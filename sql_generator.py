# sql_generator.py

import json
from typing import Any, Dict

import pandas as pd
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
            ("system", """你是一个SQL专家，根据用户的问题和slide_params生成一个有效的SQL查询语句。
        数据库表结构信息如下:
        - 表名: public.new_house, public.resale_house
        - 字段: date_code, supply_sets, trade_sets, dim_area, dim_price, dim_unit_price, city_name, district_name, block_name, project_name

        要求:
        1. WHERE子句构建: 确保包含城市、区域、板块、项目名和日期的精确匹配条件，并将年份转换为年初到年末的完整日期范围（如2021年对应'2021-01-01'到'2021-12-31'）。
        2. 输出要求: 仅返回SQL查询语句本身，不包含任何解释或代码块标记(```)。

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
            'table_name': '2020-2022年首钢供应与成交趋势', 
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
        user_question: 基于该模板，请生成2021-2023年上海市徐汇区古美板块板块的详细报告, 将面积段间隔设置为30㎡。
        slide_params:{{
            'table_name': '2020-2022年南山CBD二手房面积段房源数量统计', 
            'row_headers': ['0-20㎡', '20-40㎡', '40-60㎡', '60-80㎡', '80-100㎡', '100-120㎡', '120-140㎡', '140-160㎡', '160-180㎡', '180-200㎡', '≥200㎡'], 
            'column_headers': ['area_range', 'count']
        }}
        回答: SELECT dim_area FROM public.resale_house WHERE city_name = '上海市' AND district_name = '徐汇区' AND block_name = '古美板块' AND date_code >= '2021-01-01' AND date_code <= '2023-12-31'
        
        示例5:
        user_question: 基于该模板，请生成2020-2022年北京市密云区密云区板块的分析报告, 将面积段间隔设置为30㎡。
        slide_params:{{
            'table_name': '2020-2022年良乡价格段房源数量统计', 
            'row_headers': ['0-200万元', '200-400万元', '400-600万元', '600-800万元', '800-1000万元', '1000-1200万元', '≥1200万元'], 
            'column_headers': ['price_range', 'count']
        }}
        回答: SELECT dim_price FROM public.new_house WHERE city_name = '北京市' AND district_name = '密云区' AND block_name = '密云区' AND date_code >= '2020-01-01' AND date_code <= '2022-12-31'
        
        示例6:
        user_question: 基于该模板，请生成2020-2022年北京市密云区密云区板块的分析报告, 将价格段间隔设置为150万元。
        slide_params:{{
            'table_name': '2020-2022年良乡面积-总价交叉分析', 
            'row_headers': ['0-20㎡', '20-40㎡', '40-60㎡', '60-80㎡', '80-100㎡', '100-120㎡', '120-140㎡', '140-160㎡', '160-180㎡', '180-200㎡', '≥200㎡'], 
            'column_headers': ['0-200万元', '200-400万元', '400-600万元', '600-800万元', '800-1000万元', '1000-1200万元', '≥1200万元']
        }}
        回答: SELECT dim_area, dim_price FROM public.new_house WHERE city_name = '北京市' AND district_name = '密云区' AND block_name = '密云区' AND date_code >= '2020-01-01' AND date_code <= '2022-12-31'
        
        示例7:
        user_question: 基于该模板，请生成2021-2023年上海市徐汇区古美板块板块的分析报告。
        slide_params:{{
            'table_name': '2020-2022年良乡二手房面积段房源数量统计', 
            'row_headers': ['0-20㎡', '20-40㎡', '40-60㎡', '60-80㎡', '80-100㎡', '100-120㎡', '120-140㎡', '140-160㎡', '160-180㎡', '180-200㎡', '≥200㎡'], 
            'column_headers':  ['area_range', 'count']
        }}
        回答: SELECT dim_area FROM public.resale_house WHERE city_name = '上海市' AND district_name = '徐汇区' AND block_name = '古美板块' AND date_code >= '2021-01-01' AND date_code <= '2023-12-31'
        
        示例8:
        user_question: 基于该模板，请生成北京市怀柔区怀柔区板块的详细报告。
        slide_params:{{
            'table_name': '永丰商品住宅历年套数量', 
            'row_headers': ['2020', '2021', '2022', '2023', '2024'],,
            'column_headers':  ['year', '供应套数', '成交套数']
        }}
        回答: SELECT date_code, supply_sets, trade_sets FROM public.new_house WHERE city_name = '北京市' AND district_name = '怀柔区' AND block_name = '怀柔区' AND date_code >= '2020-01-01' AND date_code <= '2024-12-31'
      
        示例9:
        user_question: 基于该模板，请生成北京市怀柔区怀柔区板块的详细报告。
        slide_params:{{
            'table_name': '沙井商品住宅历年面积量', 
            'row_headers': ['2020', '2021', '2022', '2023', '2024'],,
            'column_headers':  ['year', '供应面积（万m2）', '成交面积（万m2）']
        }}
        回答: SELECT date_code, supply_sets, trade_sets, dim_area, dim_unit_price FROM public.new_house WHERE city_name = '北京市' AND district_name = '怀柔区' AND block_name = '怀柔区' AND date_code >= '2020-01-01' AND date_code <= '2024-12-31'
      
        示例10:
        user_question: 基于该模板，请生成2020-2024年北京市石景山区首钢板块的分析报告。
        slide_params:{{
            'table_name': '黄阁商品住宅历年市场容量', 
            'row_headers': ['2020', '2021', '2022', '2023', '2024'],,
            'column_headers':  ['year', '供应面积（万m2）', '供应套数', '成交面积（万m2）','成交套数', '成交均价（元/m2）']
        }}
        回答: SELECT date_code, supply_sets, trade_sets, dim_area, dim_unit_price FROM public.['new_house'] WHERE city_name = '北京市' AND district_name = '石景山区' AND block_name = '首钢' AND date_code >= '2020-01-01' AND date_code <= '2024-12-31'
      
        示例11:
        user_question: 基于该模板，请生成上海市浦东新区北蔡板块的分析报告。
        slide_params:{{
            'table_name': '南山CBD二手房成交套数及均价统计', 
            'row_headers': ['2020', '2021', '2022', '2023', '2024'],,
            'column_headers':  ['year', '成交面积（万m2）','成交套数', '成交均价（元/m2）']
        }}
        回答: SELECT dim_area, dim_unit_price, date_code, trade_sets FROM public.resale_house WHERE city_name = '上海市' AND district_name = '浦东新区' AND block_name = '北蔡' AND date_code >= '2020-01-01' AND date_code <= '2024-12-31'
      
        示例12:
        user_question: 基于该模板，请生成上海市浦东新区大三林板块板块的分析报告。
        slide_params:{{
            'table_name': '南山CBD二手房成交均价分布', 
            'row_headers': ['2020', '2021', '2022', '2023', '2024'],,
            'column_headers':  ['year', '成交均价（元/m2）']
        }}
        回答: SELECT dim_area, dim_unit_price, date_code, trade_sets FROM public.resale_house WHERE city_name = '上海市' AND district_name = '浦东新区' AND block_name = '大三林板块' AND date_code >= '2020-01-01' AND date_code <= '2024-12-31'
      
        示例13:
        user_question: 基于该模板，请生成上海市普陀区真如、曹杨板块的分析报告。
        slide_params:{{
            'table_name': '南山CBD二手房成交均价分布', 
            'row_headers': ['2020', '2021', '2022', '2023', '2024'],,
            'column_headers':  ['year', '成交套数']
        }}
        回答: SELECT dim_area, dim_unit_price, date_code, trade_sets FROM public.resale_house WHERE city_name = '上海市' AND district_name = '普陀区' AND block_name = '真如、曹杨' AND date_code >= '2020-01-01' AND date_code <= '2024-12-31'
      
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

        - 在解析失败时，最多重试 2 次（总共 3 次尝试）。
        - 自动去除 ```json ... ``` 代码块包裹。
        - 若最终仍失败，抛出 JSONDecodeError。
        """
        max_retries = 2  # 额外重试次数
        attempt = 0
        while attempt <= max_retries:
            try:
                chain = self.datasource_prompt_template | self.model
                response = chain.invoke({"user_question": user_question})
                json_string = response.content.strip()
                if json_string.startswith("```json"):
                    json_string = json_string[7:]
                if json_string.endswith("```"):
                    json_string = json_string[:-3]
                return json.loads(json_string)

            except json.JSONDecodeError as e:
                attempt += 1
                if attempt <= max_retries:
                    print(f"警告: 第 {attempt} 次解析失败，正在重试... 原始返回: {json_string}")
                else:
                    print(f"错误: LLM返回的JSON格式无效（已重试 {max_retries} 次仍失败）。原始返回: {json_string}")
                    raise
            except Exception as e:
                # 其他异常（如调用链失败）也进行重试
                attempt += 1
                if attempt <= max_retries:
                    print(f"警告: 第 {attempt} 次调用失败，正在重试... 错误: {e}")
                else:
                    print(f"错误: 调用链执行失败（已重试 {max_retries} 次仍失败）。错误: {e}")
                    raise

    def process_slide_params(self, slide_params: Dict[str, Any]):
        """
        这块代码暂时只实现了功能，后续会优化
        """
        title = slide_params['title']['content']
        data = slide_params['data']

        # 兼容 list / dict -> DataFrame
        if not hasattr(data, 'columns'):
            if isinstance(data, list):
                if data and isinstance(data[0], dict):
                    df = pd.DataFrame(data)
                else:
                    cols = slide_params.get('columns')
                    df = pd.DataFrame(data, columns=cols) if cols else pd.DataFrame(data)
            elif isinstance(data, dict):
                rows = data.get('rows')
                cols = data.get('columns')
                if rows is not None:
                    df = pd.DataFrame(rows, columns=cols) if cols else pd.DataFrame(rows)
                else:
                    inner = data.get('data')
                    if isinstance(inner, list) and inner and isinstance(inner[0], dict):
                        df = pd.DataFrame(inner)
                    else:
                        df = pd.DataFrame(data)  # 最后一搏
            else:
                raise TypeError(f"data 类型不受支持: {type(data)}")
        else:
            df = data

        if df.shape[1] < 1:
            raise ValueError("期望至少包含 1 列数据")

        second_col_name = df.columns[0]
        df2 = df.set_index(second_col_name)

        #列标题（不包含作为索引的那列）
        column_headers = list(df2.columns)

        #行标题（来自第二列索引）
        row_headers = list(df2.index)

        table_params = {"table_name": title,"row_headers": row_headers,"column_headers": column_headers}
        return table_params