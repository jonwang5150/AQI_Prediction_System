from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from openai import OpenAI
client = OpenAI()

class Inputs(BaseModel):
    query: str = Field(description="User query")
    country_code: str = Field(description="ISO 3166-1 alpha-2 suggested by the language of the user query")



class SearchTool(BaseTool):

    input_output_parser: PydanticOutputParser = PydanticOutputParser(pydantic_object=Inputs)
    input_format_instructions: str = input_output_parser.get_format_instructions()
    
    name:str = "Websearch-tool"
    description_template:str = dedent("""\    
    Use this tool to collect information from the internet, when you are not sure you know the answer.
    The input contains the user's question `query` and the ISO 3166-1 alpha-2 `country_code` inferred from the user's language.
    input format instructions: {input_format_instructions}
    """)

    description: str = description_template.format(input_format_instructions=input_format_instructions)
    
    def _run(self, **input):
        
        # 檢查 input 是否是 有多個輸入
        print(input)

        # 有時候會有兩種輸入結構:
        """
        input = {'input': {'query': query, 'country_code': country_code}}

        或是

        input = {'query': query, 'country_code': country_code}
        """
        # 抓取'input'這個key的值，如果沒有就返回input自己本身(對應兩種輸入結構)
        input = input.get('input', input)
        
        query = input["query"]
        country_code = input["country_code"]
        
        messages = [{"role": "user",
                     "content": query}]

        response = client.responses.create(
                    model="gpt-4o-mini",
                    tools=[
                        {"type": "web_search",
                         "user_location":{
                             "type": "approximate",
                             "country": country_code,
                         },
                        "search_context_size": "medium"
                        }],
                    tool_choice="auto",
                    input=messages)
        
        return response.output_text
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")