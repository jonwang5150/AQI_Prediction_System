import asyncio
import re
import subprocess
import sys
import uuid
from pathlib import Path
from textwrap import dedent
from typing import Optional, Type
from chat_prompt import build_standard_chat_prompt_template
from langchain.tools import BaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field


MATPLOTLIB_TOOL_SYSTEM_PROMPT = dedent("""\
你是一位資深的資料視覺化工程師，專精於使用 Python 的 matplotlib 繪製清楚、專業、可直接用於報告的圖表。

你的任務是根據使用者提供的資料與繪圖需求，產出「完整、可直接執行」的 matplotlib 程式碼。

指導原則：

1. 程式碼規範
   - 只輸出一個 ```python 程式碼區塊，區塊之外不要有任何說明文字。
   - 程式碼必須自成一體：包含所有 import 與資料定義，不得讀取外部檔案或網路資源。
   - 開頭固定使用：
       import matplotlib
       matplotlib.use("Agg")
       import matplotlib.pyplot as plt
   - 環境已預先提供變數 OUTPUT_PATH（圖檔儲存路徑），請直接使用：
       plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
     不得呼叫 plt.show()，結尾須呼叫 plt.close()。

2. 中文字型
   - 一律加入以下設定以避免中文亂碼與負號問題：
       plt.rcParams["font.sans-serif"] = ["Noto Sans CJK TC", "Microsoft JhengHei", "PingFang TC", "Arial Unicode MS", "DejaVu Sans"]
       plt.rcParams["axes.unicode_minus"] = False
   - 圖表標題、軸標籤、圖例預設使用繁體中文，除非使用者另有指定。

3. 圖表品質
   - 依資料特性選擇適當圖表類型（折線圖、長條圖、散點圖、圓餅圖、熱圖、堆疊圖、多子圖等）。
   - 必須包含：標題、軸標籤（含單位）、必要時的圖例與資料來源註記。
   - 設定合理的 figsize、字體大小、配色與格線，避免元素重疊（必要時使用 plt.tight_layout() 或 constrained layout）。
   - 數值較多時加入適當的刻度格式化（如千分位、百分比）。

4. 資料誠實性
   - 僅使用使用者提供的資料；若使用者未提供數據而要求示意圖，須在圖中以文字標註「示意資料，非實際數據」。
   - 不得捏造具體統計數值並將其呈現為事實。

5. 安全限制
   - 不得使用 os、subprocess、socket、requests、eval、exec 等與繪圖無關的模組或函式。
   - 允許使用的函式庫僅限：matplotlib、numpy、pandas、math、datetime、collections。

你不負責工具管理或對話流程控制。
除非缺乏繪圖所需的關鍵資料，否則不主動提出追問。
你的唯一職責是產出高品質、可執行的 matplotlib 繪圖程式碼。
""")


class MatplotlibToolInput(BaseModel):
    query: str = Field(
        description=(
            "繪圖需求說明，需包含：(1) 要呈現的資料或數據（可直接貼上數值）、"
            "(2) 期望的圖表類型（可省略，由工具判斷）、"
            "(3) 標題、軸標籤、單位、配色等偏好（可省略）。"
        )
    )


_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """從 LLM 回覆中取出 python 程式碼區塊；若無區塊則視整段為程式碼。"""
    match = _CODE_BLOCK_RE.search(text)
    return (match.group(1) if match else text).strip()


class MatplotlibTool(BaseTool):
    name: str = "MatplotlibTool"
    description: str = dedent("""\
一個專門產出 matplotlib 圖表的視覺化工具，可將資料與繪圖需求轉換為圖表圖檔。

當出現以下情況時應使用此工具：
- 使用者要求繪製、畫出、視覺化某組資料
- 需要折線圖、長條圖、散點圖、圓餅圖、熱圖、堆疊圖、多子圖等統計圖表
- 需要將分析結果、趨勢或比較以圖表呈現

此工具負責：
- 依資料特性選擇合適的圖表類型
- 產生完整、可執行的 matplotlib 程式碼
- 執行程式碼並輸出 PNG 圖檔（回傳圖檔路徑與原始程式碼）

輸入內容應清楚說明：
- 要繪製的資料（請直接提供數值，工具不會讀取外部檔案）
- 期望的圖表類型（可省略）
- 標題、軸標籤、單位、配色等細節（可省略）

此工具僅負責圖表繪製，不負責資料分析、資料蒐集或對話管理。
""")
    args_schema: Type[BaseModel] = MatplotlibToolInput

    runnable: Runnable
    output_dir: Path = Path("./charts")
    execute: bool = True  # False 時僅回傳程式碼，不執行
    timeout: int = 60

    @classmethod
    def create(
        cls,
        llm: Runnable,
        output_dir: str | Path = "./charts",
        execute: bool = True,
        timeout: int = 60,
    ):
        # build_standard_chat_prompt_template 沿用你原本專案中的函式
        input_ = {
            "system": {"template": MATPLOTLIB_TOOL_SYSTEM_PROMPT},
            "human": {"template": "{user_query}", "input_variable": ["user_query"]},
        }
        chat_prompt_template = build_standard_chat_prompt_template(input_)  # noqa: F821

        pipeline = chat_prompt_template | llm | StrOutputParser()

        return cls(
            runnable=pipeline,
            output_dir=Path(output_dir),
            execute=execute,
            timeout=timeout,
        )

    # ---------- 執行繪圖程式碼 ----------
    def _execute_code(self, code: str) -> tuple[Optional[Path], Optional[str]]:
        """在獨立子行程中執行程式碼，回傳 (圖檔路徑, 錯誤訊息)。"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        run_id = uuid.uuid4().hex[:8]
        img_path = (self.output_dir / f"chart_{run_id}.png").resolve()
        script_path = self.output_dir / f"chart_{run_id}.py"

        script_path.write_text(
            f"OUTPUT_PATH = {str(img_path)!r}\n{code}\n", encoding="utf-8"
        )

        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return None, f"程式碼執行超過 {self.timeout} 秒，已中止。"

        if proc.returncode != 0:
            return None, proc.stderr.strip()[-1500:]
        if not img_path.exists():
            return None, "程式碼執行完成，但未產生圖檔（請確認有呼叫 plt.savefig(OUTPUT_PATH)）。"
        return img_path, None

    # ---------- LangChain 介面 ----------
    def _run(self, query: str) -> str:
        raw = self.runnable.invoke({"user_query": query})
        code = extract_code(raw)

        if not self.execute:
            return f"以下為 matplotlib 程式碼：\n```python\n{code}\n```"

        img_path, error = self._execute_code(code)

        # 執行失敗時，把錯誤回饋給 LLM 重試一次
        if error:
            retry_query = (
                f"{query}\n\n"
                f"你先前產生的程式碼執行失敗，請修正後重新輸出完整程式碼。\n"
                f"【先前程式碼】\n```python\n{code}\n```\n【錯誤訊息】\n{error}"
            )
            raw = self.runnable.invoke({"user_query": retry_query})
            code = extract_code(raw)
            img_path, error = self._execute_code(code)

        if error:
            return f"圖表產生失敗：\n{error}\n\n最後一版程式碼：\n```python\n{code}\n```"

        return f"圖表已產生，檔案路徑：{img_path}\n\n繪圖程式碼：\n```python\n{code}\n```"

    async def _arun(self, query: str) -> str:
        return await asyncio.to_thread(self._run, query)