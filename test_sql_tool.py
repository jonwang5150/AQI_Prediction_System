"""SQL agent 的查詢邊界測試，不連線資料庫或模型。"""
import json
from datetime import datetime
import unittest
from unittest.mock import MagicMock
from langchain_core.runnables import RunnableLambda

from pydantic import ValidationError
from sqlalchemy.dialects import postgresql

from agent.sql_tool import QueryPlan, QuerySpec, SQLTool, TABLES, build_select, execute_plan


class SQLToolTests(unittest.TestCase):
    def test_only_three_tables(self):
        for table in TABLES:
            sql = str(build_select(QuerySpec(table=table)).compile(dialect=postgresql.dialect()))
            self.assertTrue(sql.startswith('SELECT '))
            self.assertIn('FROM public.' + table, sql)
        for table in ['pg_authid', 'information_schema.tables', 'air_quality_hourly; DROP TABLE x']:
            with self.assertRaises(ValidationError):
                QuerySpec(table=table)

    def test_no_raw_sql_or_functions(self):
        for extra in [{'sql': 'DELETE FROM air_quality_hourly'},
                      {'metrics': [{'function': 'pg_read_file', 'column': 'aqi'}]},
                      {'limit': 501}]:
            with self.assertRaises(ValidationError):
                QuerySpec(table='air_quality_hourly', **extra)
        for extra in [{'columns': ['aqi; DROP TABLE x']}, {'order_by': 'pg_sleep(10)'},
                      {'group_by': ['pg_catalog.pg_authid'], 'metrics': [{'function': 'count', 'column': '*'}]},
                      {'filters': [{'column': 'secret', 'operator': 'eq', 'value': 1}]}]:
            with self.assertRaises(ValueError):
                build_select(QuerySpec(table='air_quality_hourly', **extra))

    def test_values_are_parameters(self):
        attack = "x'; DELETE FROM air_quality_hourly; --"
        statement = build_select(QuerySpec(table='air_quality_hourly', filters=[
            {'column': 'station', 'operator': 'eq', 'value': attack}]))
        compiled = statement.compile(dialect=postgresql.dialect())
        self.assertNotIn(attack, str(compiled))
        self.assertIn(attack, compiled.params.values())

    def test_grouped_statistics(self):
        statement = build_select(QuerySpec(table='air_quality_hourly', group_by=['station'],
            metrics=[{'function': 'avg', 'column': 'aqi'}, {'function': 'count', 'column': '*'}],
            order_by='avg_aqi', descending=True))
        sql = str(statement.compile(dialect=postgresql.dialect()))
        self.assertIn('GROUP BY public.air_quality_hourly.station', sql)
        self.assertIn('count(*) AS count_all', sql)
        self.assertIn('ORDER BY avg_aqi DESC', sql)

    def test_timestamp_filter_binds_datetime(self):
        for text in ['2026-08-01', '2026-08-01T00:00:00+08:00', '2026-07-31T16:00:00Z']:
            statement = build_select(QuerySpec(table='air_quality_hourly', filters=[
                {'column': 'observed_at', 'operator': 'ge', 'value': text}]))
            compiled = statement.compile(dialect=postgresql.dialect())
            self.assertIn(datetime(2026, 8, 1), compiled.params.values())
        with self.assertRaises(ValueError):
            build_select(QuerySpec(table='air_quality_hourly', filters=[
                {'column': 'observed_at', 'operator': 'ge', 'value': '2026-02-30'}]))

    def test_muzha_alias_and_single_day_average(self):
        for station in ('木柵', '木柵站', '木柵觀測站'):
            statement = build_select(QuerySpec(table='air_quality_hourly', filters=[
                {'column': 'station', 'operator': 'eq', 'value': station},
                {'column': 'observed_at', 'operator': 'ge', 'value': '2026-04-06'},
                {'column': 'observed_at', 'operator': 'lt', 'value': '2026-04-07'}],
                metrics=[{'function': 'avg', 'column': 'aqi'}, {'function': 'count', 'column': 'aqi'}]))
            compiled = statement.compile(dialect=postgresql.dialect())
            self.assertIn('木柵站', compiled.params.values())
            self.assertIn('avg(', str(compiled))
            self.assertIn('count(', str(compiled))
            self.assertNotIn('GROUP BY', str(compiled))

    def engine(self):
        engine = MagicMock()
        engine.dialect.name = 'postgresql'
        connection = engine.connect.return_value.execution_options.return_value.__enter__.return_value
        connection.execute.return_value.mappings.return_value.fetchall.return_value = [{'aqi': 1}, {'aqi': 2}]
        return engine, connection

    def test_read_only_before_select_and_rollback(self):
        engine, connection = self.engine()
        plan = QueryPlan(queries=[QuerySpec(table='air_quality_hourly', columns=['aqi'], limit=1)])
        result = execute_plan(engine, plan)
        self.assertEqual(result['results'][0]['rows'], [{'aqi': 1}])
        self.assertTrue(result['results'][0]['truncated'])
        calls = connection.method_calls
        self.assertEqual(calls[1].args, ('SET TRANSACTION READ ONLY',))
        self.assertLess(next(i for i, c in enumerate(calls) if c[0] == 'exec_driver_sql'),
                        next(i for i, c in enumerate(calls) if c[0] == 'execute'))
        connection.begin.return_value.rollback.assert_called_once()
        connection.execute.side_effect = RuntimeError('failure')
        with self.assertRaises(RuntimeError):
            execute_plan(engine, plan)
        self.assertEqual(connection.begin.return_value.rollback.call_count, 2)

    def test_invalid_plan_never_connects(self):
        engine, _ = self.engine()
        plan = QueryPlan(queries=[QuerySpec(table='air_quality_hourly'),
            QuerySpec(table='air_quality_hourly', columns=['secret'])])
        with self.assertRaises(ValueError):
            execute_plan(engine, plan)
        engine.connect.assert_not_called()
        execute_plan(engine, QueryPlan(refusal='不允許寫入'))
        engine.connect.assert_not_called()

    def test_tool_interface(self):
        engine, _ = self.engine()
        llm = MagicMock()
        response = {'queries': [{'table': 'air_quality_predictions', 'limit': 1}]}
        llm.with_structured_output.return_value = RunnableLambda(lambda _: response)
        tool = SQLTool.create(llm=llm, engine=engine)
        llm.with_structured_output.assert_called_once_with(QueryPlan, method='function_calling')
        output = json.loads(tool.invoke({'query': '查詢預測'}))
        self.assertEqual(output['results'][0]['table'], 'air_quality_predictions')
        response = {'sql': 'DROP TABLE x'}
        self.assertIn('error', json.loads(tool.invoke({'query': '刪除資料'})))

    def test_invalid_model_plan_repaired_once(self):
        engine, _ = self.engine()
        outputs = iter([{'queries': [{'table': 'public.air_quality_hourly'}]},
                        {'queries': [{'table': 'air_quality_hourly', 'columns': ['aqi']}]}])
        tool = SQLTool(runnable=RunnableLambda(lambda _: next(outputs)), engine=engine)
        result = json.loads(tool.invoke({'query': 'test'}))
        self.assertIn('results', result)

    def test_invalid_model_plan_stops_after_two_attempts(self):
        engine, _ = self.engine()
        calls = []
        def malformed(value):
            calls.append(value)
            return None
        tool = SQLTool(runnable=RunnableLambda(malformed), engine=engine)
        result = json.loads(tool.invoke({'query': 'test'}))
        self.assertEqual(len(calls), 2)
        self.assertEqual(result['error_code'], 'invalid_query_plan')
        self.assertFalse(result['retryable'])
        engine.connect.assert_not_called()


if __name__ == '__main__':
    unittest.main()
