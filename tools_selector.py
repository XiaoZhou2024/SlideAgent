from pathlib import Path
from typing import Dict
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

from file_utils import load_prompt_from_file
from tool_functions import *
from config import config

class ToolSelector:
    def __init__(self):

        self.model = init_chat_model(
            base_url=config.BASE_URL, 
            api_key=config.API_KEY, 
            model=config.MODEL_NAME, 
            model_provider="openai"
        )
        self.agent = create_react_agent(
                    model=self.model,
                    tools=[
                        execute_analysis
                    ],
        )

    def _count_csv_files(self, dir_path: str | Path) -> int:
        p = Path(dir_path)
        return sum(1 for _ in p.glob("*.csv"))
    def select_function_by_intent(self, query_filters: Dict, update_filters: list, data_path: Path):
        base_path = Path(data_path)
        retrieval_path = base_path / "retrieval"
        processed_path = base_path / "processed"
        processed_path.mkdir(parents=True, exist_ok=True)

        for i, update in enumerate(update_filters):
            params = {
                "options_json": json.dumps(update["fun_tool"]["quadruples"], ensure_ascii=False),
                "input_path": str(retrieval_path / f"{i}.csv").replace("\\", "\\\\"),
                "output_path": str(processed_path / f"{i}.xlsx").replace("\\", "\\\\"),
                'price_range_size': 1 if query_filters.get("price_range_size") == 'default' else query_filters.get(
                    "price_range_size"),
                'area_range_size': 20 if query_filters.get("area_range_size") == 'default' else query_filters.get(
                    "area_range_size"),
            }
            print(params)
            prompt = load_prompt_from_file("tool_prompt.txt").format(
            input_path=params['input_path'],
            output_path=params['output_path'],
            price_range_size=params['price_range_size'],
            area_range_size=params['area_range_size'],
            options=params['options_json']
            )

            res = self.agent.invoke({
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }, {"recursion_limit": 10})


            ai_msg = next(m for m in res["messages"] if m.type == "ai")
            tool_calls = getattr(ai_msg, "tool_calls", [])
            for call in tool_calls:
                print(call['name'])
                print(call["args"])
                print(update_filters)
                if "options" in call.get("args", {}):
                    update_filters[i]['fun_tool']['args']["options"] = call["args"]["options"]
                if "area_range_size" in call.get("args", {}):
                    update_filters[i]['fun_tool']['args']["area_range_size"] = call["args"]["area_range_size"]
                if "price_range_size" in call.get("args", {}):
                    update_filters[i]['fun_tool']['args']["price_range_size"] = call["args"]["price_range_size"]

        return update_filters
