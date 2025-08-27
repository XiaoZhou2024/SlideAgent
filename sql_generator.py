# sql_generator.py

import json
from typing import Any, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config import config

class SqlGenerator:
    """
    使用大语言模型根据用户问题生成SQL查询的类。
    """
    def __init__(self, temperature: float = 0):
        """
        初始化SQL生成器。

        Args:
            api_key (str): DeepSeek API密钥。
            model_name (str, optional): 使用的模型名称. 默认为 "deepseek-chat".
            temperature (float, optional): 模型温度参数. 默认为 0.
        """

        self.model = ChatOpenAI(
            base_url=config.BASE_URL, 
            api_key=config.API_KEY, 
            temperature=temperature, 
            model=config.MODEL_NAME
        )
        self.sql_prompt_template = self._create_sql_prompt_template()
        self.datasource_prompt_template = self._create_datasource_prompt_template()

    def _create_sql_prompt_template(self) -> ChatPromptTemplate:
        """创建用于生成SQL的Prompt模板。"""
        return ChatPromptTemplate.from_messages([
            ("system", """你是一个SQL专家，根据用户的问题生成一个有效的SQL查询语句。
        数据库表结构信息如下:
        - 表名: public.new_house, public.resale_house
        - 字段: supply_sets, trade_sets, dim_area, dim_price, dim_unit_price, city_name, district_name, block_name, project_name, date_code

        要求:
        1. 只生成SQL查询语句本身，不要包含任何解释或代码块标记(```)。
        2. 根据问题中的城市、区域、板块和年份来构建WHERE子句。
        3. 日期范围应为年初到年末，例如2021年应为 '2021-01-01' 到 '2021-12-31'。

        示例1:
        user_question: 基于该模板，请生成2020-2022年北京市怀柔区怀柔区板块的分析报告.
        slide_params:{{
            'table_name': '2020-2022年良乡供应与成交套数及占比', 
            'row_headers': ['0-20㎡', '20-40㎡', '40-60㎡', '60-80㎡', '80-100㎡', '100-120㎡', '120-140㎡', '140-160㎡', '160-180㎡', '180-200㎡', '200-220㎡', '220-240㎡', '240-260㎡', '260-280㎡', '280-300㎡', '≥300㎡', '总计'], 
            'column_headers': ['供应套数', '成交套数', '供求比', '成交占比']
        }}
        回答: SELECT supply_sets, trade_sets, dim_area FROM public.new_house WHERE city_name = '北京市' AND district_name = '怀柔区' AND block_name = '怀柔区' AND date_code >= '2020-01-01' AND date_code <= '2022-12-31'

        示例2:
        user_question: 基于该模板，请生成2020-2022年北京市海淀区永丰板块的分析报告, 将面积段间隔设置为15㎡。
        slide_params:{{
            'table_name': '2020-2022年良乡供应与成交趋势', 
            'row_headers': ['0-20㎡', '20-40㎡', '40-60㎡', '60-80㎡', '80-100㎡', '100-120㎡', '120-140㎡', '140-160㎡', '160-180㎡', '180-200㎡', '200-220㎡', '220-240㎡', '240-260㎡', '260-280㎡', '280-300㎡', '≥300㎡'], 
            'column_headers': ['2020供应套数', '2020成交套数', '2021供应套数', '2021成交套数', '2022供应套数', '2022成交套数']
        }}
        回答: SELECT date_code, supply_sets, trade_sets, dim_area FROM public.new_house WHERE city_name = '北京市' AND district_name = '海淀区' AND block_name = '永丰' AND date_code >= '2020-01-01' AND date_code <= '2022-12-31'
        
        示例3:
        user_question: 基于该模板，请生成2024年1-6月北京市房山区良乡中建学府印悦二期的分析报告。
        slide_params:{{
            'table_name': '2024年1-6月密云区国祥府房价走势', 
            'row_headers': ['2024-01', '2024-02', '2024-03', '2024-04', '2024-05', '2024-06'], 
            'column_headers': ['date', 'avg_unit_price']
        }}
        回答: SELECT date_code, dim_unit_price FROM public.new_house WHERE city_name = '北京市' AND district_name = '房山区' AND block_name = '良乡' AND project_name ='中建学府印悦二期' AND date_code >= '2024-01-01' AND date_code <= '2024-06-30'
        
        示例4:
        user_question: 基于该模板，请生成2021-2023年上海市徐汇区古美板块板块的详细报告, 将面积段间隔设置为30㎡.
        slide_params:{{
            'table_name': '2020-2022年南山CBD二手房面积段房源数量统计', 
            'row_headers': ['0-20㎡', '20-40㎡', '40-60㎡', '60-80㎡', '80-100㎡', '100-120㎡', '120-140㎡', '140-160㎡', '160-180㎡', '180-200㎡', '≥200㎡'], 
            'column_headers': ['area_range', 'count']
        }}
        回答: SELECT dim_area FROM public.resale_house WHERE city_name = '上海市' AND district_name = '徐汇区' AND block_name = '古美板块' AND date_code >= '2021-01-01' AND date_code <= '2023-12-31'
        
        """),

            ("human", "user_question:{user_question}  slide_params:{slide_params}")
        ])
    
    def _create_datasource_prompt_template(self) -> ChatPromptTemplate:
        """创建用于生成数据源JSON的Prompt模板。"""
        # 将示例JSON中的花括号转义为双花括号 {{ 和 }}
        return ChatPromptTemplate.from_messages([
            ("system", """你是一个数据提取专家。根据用户的问题，提取城市、区、板块、时间范围、，并生成一个JSON对象。
            要求:
            1. 数据库名称固定为 "new_house"。
            2. 严格按照JSON格式输出，不要包含任何解释或代码块标记。
            3. 时间格式为 'YYYY-MM-DD'。

            示例:
            问题: 基于该模板，请生成2021-2023年北京市怀柔区怀柔区板块的详细报告
            回答:
            {{
            "city": "北京市",
            "district": "怀柔区",
            "block": "怀柔区",
            "project": "default",
            "start_time": "2021-01-01",
            "end_time": "2023-12-31",
            "area_range_size": "default",
            "price_range_size": "default"
            }}

            问题: 基于该模板，请生成2020-2022年深圳市龙岗区龙岗中心城板块的分析报告, 将面积段间隔设置为25㎡, 将价格段间隔设置为200万元。
            回答:
            {{
            "city": "深圳市",
            "district": "龙岗区",
            "block": "龙岗中心城",
            "project": "default",
            "start_time": "2020-01-01",
            "end_time": "2022-12-31",
            "area_range_size": "25",
            "price_range_size": "200"
            }}
            
            问题: 基于该模板，请生成2020-2022年广州市南沙区黄阁板块的分析报告, 将价格段间隔设置为100万元。
            回答:
            {{
            "city": "广州市",
            "district": "南沙区",
            "block": "黄阁",
            "project": "default",
            "start_time": "2020-01-01",
            "end_time": "2022-12-31",
            "area_range_size": "default",
            "price_range_size": "100"
            }}
            
            问题: 基于该模板，请生成2024年1-6月北京市房山区良乡中建学府印悦二期的分析报告。
            回答:
            {{
            "city": "北京市",
            "district": "房山区",
            "block": "良乡",
            "project": "中建学府印悦二期",
            "start_time": "2024-1-01",
            "end_time": "2024-6-31",
            "area_range_size": "default",
            "price_range_size": "default"
            }}
            """),
            ("human", "{user_question}")
        ])

    def generate_sql(self, user_question: str, slide_params: Dict[str, Any]) -> list:
        """
        根据用户问题生成SQL查询。

        Args:
            user_question (str): 用户的报告需求或问题。

        Returns:
            str: 生成的SQL查询语句。
        """
        chain = self.sql_prompt_template | self.model
        slide_tables = slide_params['template_slide']['content_elements']
        sql_list = []
        for slide_table_params in slide_tables:
            table_params = self.process_slide_params(slide_table_params)
            response = chain.invoke({"user_question": user_question, "slide_params": table_params})

            sql_query = response.content.strip()
            if sql_query.startswith("```sql"):
                sql_query = sql_query[6:]
            if sql_query.endswith("```"):
                sql_query = sql_query[:-3]
            sql_list.append(sql_query.strip())
        
        return sql_list

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

    def process_slide_params(self, slide_params: Dict[str, Any]):
        """
        这块代码暂时只实现了功能，后续会优化
        """
        title = slide_params['title']['content']
        df = slide_params['data']
        second_col_name = df.columns[0]
        df2 = df.set_index(second_col_name)

        #列标题（不包含作为索引的那列）
        column_headers = list(df2.columns)

        #行标题（来自第二列索引）
        row_headers = list(df2.index)

        table_params = {"table_name": title,"row_headers": row_headers,"column_headers": column_headers}
        return table_params