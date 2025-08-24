# from typing import Dict, Any
#
# from langgraph.prebuilt import create_react_agent
# from langchain.chat_models import init_chat_model
# from langgraph.prebuilt import create_react_agent
# from data_process_tool import *
#
# class ToolSelector:
#     """
#     使用大语言模型根据用户问题生成SQL查询的类。
#     """
#     def __init__(self, base_url: str, api_key: str, model_name: str, ):
#         """
#         初始化SQL生成器。
#
#         Args:
#             api_key (str): DeepSeek API密钥。
#             model_name (str, optional): 使用的模型名称. 默认为 "deepseek-chat".
#             temperature (float, optional): 模型温度参数. 默认为 0.
#         """
#
#         self.model = init_chat_model(base_url=base_url, api_key=api_key, model=model_name, model_provider="openai")
#         self.agent = create_react_agent(
#                     model=self.model,
#                     tools=[
#                         supply_and_sales_counts_and_share,
#                         analyze_supply_sales_trend,
#                         get_supply_sales_counts_stats,
#                         compute_area_price_cross_stats,
#                         compute_area_num_stats,
#                         compute_price_num_stats,
#                         compute_market_capacity,
#                         compute_annual_traded_units,
#                         compute_annual_traded_area,
#                         compute_resale_house_total_and_avg_price,
#                         compute_resale_house_transaction_count_distribution,
#                         compute_resale_house_avg_price_distribution,
#                         get_recent_transaction_trend,
#                         compute_market_capacity
#                     ],
#         )
#
#         def select_function_by_intent(self, user_question: str,slide_params: Dict[str, Any]):
#
