import json
from pathlib import Path
from typing import Dict, Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from data_process_tool import *
from config import config

class ToolSelector:
    """
    使用大语言模型根据用户问题生成SQL查询的类。
    """
    def __init__(self):
        """
        初始化SQL生成器。

        Args:
            api_key (str): DeepSeek API密钥。
            model_name (str, optional): 使用的模型名称. 默认为 "deepseek-chat".
            temperature (float, optional): 模型温度参数. 默认为 0.
        """

        self.model = init_chat_model(
            base_url=config.BASE_URL, 
            api_key=config.API_KEY, 
            model=config.MODEL_NAME, 
            model_provider="openai"
        )
        self.agent = create_react_agent(
                    model=self.model,
                    tools=[
                        supply_and_sales_counts_and_share,
                        analyze_supply_sales_trend,
                        get_supply_sales_counts_stats,
                        compute_area_price_cross_stats,
                        compute_area_num_stats,
                        compute_price_num_stats,
                        compute_market_capacity,
                        compute_annual_traded_units,
                        compute_annual_traded_area,
                        compute_resale_house_total_and_avg_price,
                        compute_resale_house_transaction_count_distribution,
                        compute_resale_house_avg_price_distribution,
                        get_recent_transaction_trend,
                        compute_market_capacity
                    ],
        )

    def _count_csv_files(self, dir_path: str | Path) -> int:
        p = Path(dir_path)
        return sum(1 for _ in p.glob("*.csv"))
    def select_function_by_intent(self, data_source, slide_params: Dict[str, Any], data_path: Path):
        tool_call_params = []
        base_path = Path(data_path)
        retrieval_path = base_path / "retrieval"
        processed_path = base_path / "processed"
        processed_path.mkdir(parents=True, exist_ok=True)
        n = self._count_csv_files(retrieval_path)
        for i in range(n):
            tool_params = {
                'tool': '',
                'args': {
                    'area_range_size': 'default',
                    'price_range_size': 'default',
                }
            }
            intent = slide_params['template_slide']['content_elements'][i]['title']['content']
            messages = [
                SystemMessage(content="""
                    你是一个意图提取专家。根据用户的问题，提取用户的关键意图。
                    参考下面案例:
                    示例1:
                    问题: 密云区供应与成交套数及占比
                    回答: 供应与成交套数及占比
                    示例2:
                    问题: 良乡供应与成交趋势
                    回答: 供应与成交趋势
                    示例3:
                    问题: 密云区国祥府房价走势
                    回答: 房价走势
                    示例4:
                    问题: 永丰商品住宅历年套数量
                    回答: 商品住宅历年套数量
            """),
                HumanMessage(content=intent),
            ]
            intent = self.model.invoke(messages).content
            print("intent是:{}".format(intent))

            params = {
                'intent': intent ,
                'project_name': data_source['project'],
                'area_range_size':data_source['area_range_size'],
                'price_range_size':data_source['price_range_size'],
                "input_path": str(retrieval_path    / f"{i}.csv"),
                "output_path": str(processed_path / f"{i}.xlsx"),
            }
            prompt = (
                    "请根据下面的参数，选择合适的工具去处理 input_path 对应的数据：\n注意：有些参数用不到可以不用使用\n"
                    + json.dumps(params, ensure_ascii=False, indent=2, default=str)
            )
            res = self.agent.invoke({
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            })
            print(res)
            ai_msg = next(m for m in res["messages"] if m.type == "ai")
            tool_calls = getattr(ai_msg, "tool_calls", [])
            for call in tool_calls:
                print(f"  -> 执行的工具是:")
                print(call['name'])
                print(f"  -> 工具的参数是:")
                print(call["args"])

                tool_params['tool'] = call['name']
                if "area_range_size" in call.get("args", {}):
                    tool_params['args']['area_range_size'] = call["args"]["area_range_size"]
                if "price_range_size" in call.get("args", {}):
                    tool_params['args']['price_range_size'] = call["args"]["price_range_size"]

            tool_call_params.append(tool_params)

        return tool_call_params
