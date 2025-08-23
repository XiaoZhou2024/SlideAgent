# sql_generator.py

import os
import json
from typing import Any, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek

class SqlGenerator:
    """
    使用大语言模型根据用户问题生成SQL查询的类。
    """
    def __init__(self, api_key: str, model_name: str = "deepseek-chat", temperature: float = 0):
        """
        初始化SQL生成器。

        Args:
            api_key (str): DeepSeek API密钥。
            model_name (str, optional): 使用的模型名称. 默认为 "deepseek-chat".
            temperature (float, optional): 模型温度参数. 默认为 0.
        """
        if not api_key:
            raise ValueError("DeepSeek API key is required.")
        os.environ["DEEPSEEK_API_KEY"] = api_key
        
        self.model = ChatDeepSeek(temperature=temperature, model=model_name)
        self.sql_prompt_template = self._create_sql_prompt_template()
        self.datasource_prompt_template = self._create_datasource_prompt_template()

    def _create_sql_prompt_template(self) -> ChatPromptTemplate:
        """创建用于生成SQL的Prompt模板。"""
        return ChatPromptTemplate.from_messages([
            ("system", """你是一个SQL专家，根据用户的问题生成一个有效的SQL查询语句。
        数据库表结构信息如下:
        - 表名: public.new_house
        - 字段: supply_sets, trade_sets, city_name, district_name, block_name, date_code

        要求:
        1. 只生成SQL查询语句本身，不要包含任何解释或代码块标记(```)。
        2. 根据问题中的城市、区域、板块和年份来构建WHERE子句。
        3. 日期范围应为年初到年末，例如2021年应为 '2021-01-01' 到 '2021-12-31'。

        示例:
        问题: 基于该模板，请生成2021-2023年北京市怀柔区怀柔区板块的详细报告
        回答: SELECT supply_sets, trade_sets FROM public.new_house WHERE city_name = '北京市' AND district_name = '怀柔区' AND block_name = '怀柔区' AND date_code >= '2021-01-01' AND date_code <= '2023-12-31'

        问题: 基于该模板，请生成2020-2022年深圳市龙岗区龙岗中心城板块的分析报告。
        回答: SELECT supply_sets, trade_sets FROM public.new_house WHERE city_name = '深圳市' AND district_name = '龙岗区' AND block_name = '龙岗中心城' AND date_code >= '2020-01-01' AND date_code <= '2022-12-31'
        """),
            ("human", "{user_question}")
        ])
    
    def _create_datasource_prompt_template(self) -> ChatPromptTemplate:
        """创建用于生成数据源JSON的Prompt模板。"""
        # 将示例JSON中的花括号转义为双花括号 {{ 和 }}
        return ChatPromptTemplate.from_messages([
            ("system", """你是一个数据提取专家。根据用户的问题，提取城市、区、板块和时间范围，并生成一个JSON对象。
            要求:
            1. 数据库名称固定为 "new_house"。
            2. 严格按照JSON格式输出，不要包含任何解释或代码块标记。
            3. 时间格式为 'YYYY-MM-DD'。

            示例:
            问题: 基于该模板，请生成2021-2023年北京市怀柔区怀柔区板块的详细报告
            回答:
            {{
            "db_name": "new_house",
            "city": "北京市",
            "district": "怀柔区",
            "block": "怀柔区",
            "start_time": "2021-01-01",
            "end_time": "2023-12-31"
            }}

            问题: 基于该模板，请生成2020-2022年深圳市龙岗区龙岗中心城板块的分析报告。
            回答:
            {{
            "db_name": "new_house",
            "city": "深圳市",
            "district": "龙岗区",
            "block": "龙岗中心城",
            "start_time": "2020-01-01",
            "end_time": "2022-12-31"
            }}
            """),
            ("human", "{user_question}")
        ])

    def generate_sql(self, user_question: str) -> str:
        """
        根据用户问题生成SQL查询。

        Args:
            user_question (str): 用户的报告需求或问题。

        Returns:
            str: 生成的SQL查询语句。
        """
        chain = self.sql_prompt_template | self.model
        response = chain.invoke({"user_question": user_question})
        
        sql_query = response.content.strip()
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]
        
        return sql_query.strip()

    def generate_datasource_json(self, user_question: str) -> Dict[str, Any]:
        """
        根据用户问题生成数据源JSON对象。

        Args:
            user_question (str): 用户的报告需求或问题。

        Returns:
            Dict[str, Any]: 代表数据源的Python字典。
        """
        chain = self.datasource_prompt_template | self.model
        response = chain.invoke({"user_question": user_question})
        
        json_string = response.content.strip()
        if json_string.startswith("```json"):
            json_string = json_string[7:]
        if json_string.endswith("```"):
            json_string = json_string[:-3]
            
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            print(f"错误: LLM返回的JSON格式无效: {json_string}")
            raise