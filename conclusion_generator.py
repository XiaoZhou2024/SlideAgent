# sql_generator.py

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config import config 

class ConclusionGenerator:
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
        self.conclusion_prompt_template = self._create_conclusion_prompt_template()
        
    def _create_conclusion_prompt_template(self) -> ChatPromptTemplate:
        """创建用于生成数据源JSON的Prompt模板。"""
        # 将示例JSON中的花括号转义为双花括号 {{ 和 }}
        return ChatPromptTemplate.from_messages(
            ("system", """你是一个结论生成专家。你需要参考给定的模板数据和模板结论，给新的数据生成结论。
            1.注意替换地点，数字等关键信息
            示例:
            template_data:
                2020-2022年良乡供应与成交套数及占比
                面积段	供应套数	成交套数	供求比	成交占比
                0-15㎡	396	181	2.19	5.1%
                15-30㎡	255	46	5.54	1.3%
                30-45㎡	74	19	3.89	0.5%
                45-60㎡	155	36	4.31	1.0%
                60-75㎡	492	382	1.29	10.8%
                75-90㎡	1563	1662	0.94	47.0%
                90-105㎡	584	432	1.35	12.2%
                105-120㎡	240	190	1.26	5.4%
                120-135㎡	56	102	0.55	2.9%
                135-150㎡	128	303	0.42	8.6%
                150-165㎡	3	6	0.5	0.2%
                165-180㎡	4	34	0.12	1.0%
                180-195㎡	5	50	0.1	1.4%
                195-210㎡	2	8	0.25	0.2%
                210-225㎡	2	0	0.0	0.0%
                ≥225㎡	18	82	0.22	2.3%
                总计	3977	3533	1.1	100%
            template_conclusion:
            2020-2022年良乡主力供求面积为75-90㎡，占比近47.0%，改善户型主力为180-195㎡
            
            data:
                2021-2023年中新镇供应与成交套数及占比
                面积段	供应套数	成交套数	供求比	成交占比
                0-25㎡	46	31	1.48	0.3%
                25-50㎡	254	104	2.44	1.0%
                50-75㎡	1641	1264	1.3	12.0%
                75-100㎡	7482	6242	1.2	59.3%
                100-125㎡	2884	2698	1.07	25.6%
                125-150㎡	142	178	0.8	1.7%
                150-175㎡	2	1	2.0	0.0%
                175-200㎡	1	0	0.0	0.0%
                200-225㎡	2	1	2.0	0.0%
                225-250㎡	1	0	0.0	0.0%
                250-275㎡	1	1	1.0	0.0%
                275-300㎡	0	0	0.0	0.0%
                300-325㎡	0	0	0.0	0.0%
                325-350㎡	0	0	0.0	0.0%
                350-375㎡	1	1	1.0	0.0%
                ≥375㎡	2	0	0.0	0.0%
                总计	12459	10521	1.2	100%
            conclusion:
            2021-2023年中新镇主力供求面积为75-100㎡，占比近59.3%，改善户型主力为350-375㎡



            """),
            ("human", """template_data:
                        {template_table_title}
                        {template_data}
                        template_conclusion:    
                        {template_conclusion}
                        
                        data:
                        {data}
                        conclusion:

                            """)
        ])

    def get_conclusion(self, slide_params: Dict[str, Any], data_path: Path):
        template_data = slide_params['template_slide']['content_elements'][0]['data']
        template_table_title = slide_params['template_slide']['content_elements'][0]['title']['content']
        template_conclusion = slide_params['template_slide']['analysis']['content']
        data = pd.read_excel(data_path / 'processed'/ '1.xlsx')
        chain = self.conclusion_prompt_template | self.model
        response = chain.invoke({"template_data": template_data, "template_table_title": template_table_title, "template_conclusion": template_conclusion, "data": data})
        return response.content.replace('*', '')