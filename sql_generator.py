import re
import json
import pandas as pd
from math import hypot
from config import config
from copy import deepcopy
from typing import Any, Dict, List
from langchain_openai import ChatOpenAI
from file_utils import load_prompt_from_file
from langchain_core.prompts import ChatPromptTemplate

class SqlGenerator:
    """
    SQL Generator class that uses large language models to generate SQL queries based on user questions.
    """
    def __init__(self, temperature: float = 0):
        """
        Initializes the SQL Generator.

        Args:
            temperature (float, optional): Model temperature parameter. Defaults to 0.
        """
        self.model = ChatOpenAI(
            base_url=config.BASE_URL,
            api_key=config.API_KEY,
            temperature=temperature,
            model=config.MODEL_NAME
        )
        self.sql_prompt_template = self._create_sql_prompt_template()
        self.query_filters_prompt_template = self._create_query_filters_prompt_template()
        self.slide_filters_prompt_template = self._create_slide_filters_prompt_template()

    def _create_sql_prompt_template(self) -> ChatPromptTemplate:
        system_prompt = load_prompt_from_file("generate_sql_prompt.txt")
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "slide_params:{slide_params}")
        ])

    def _create_query_filters_prompt_template(self) -> ChatPromptTemplate:
        system_prompt = load_prompt_from_file("query_filters_prompt.txt")
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{user_question}")
        ])

    def _create_slide_filters_prompt_template(self) -> ChatPromptTemplate:
        system_prompt = load_prompt_from_file("slide_filters_prompt.txt")
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{slide_params}")
        ])

    def generate_sql(self, update_filters: List):
        """
        Generates SQL queries based on the provided filter parameters.

        Args:
            update_filters (List): A list of filter configurations containing connection,
                                  select columns, and filter conditions for SQL generation.

        Returns:
            List: A list of generated SQL queries in JSON format.
        """
        chain = self.sql_prompt_template | self.model
        sql_list = []

        for i, filter in enumerate(update_filters):
            slide_params = {
                "connection": filter["connection"],
                "select_columns": filter["select_columns"],
                "filters": filter["filters"]
            }
            print(f"  -> slide_params: {slide_params}")
            response = chain.invoke({"slide_params": slide_params})

            sql_query = response.content.strip()
            sql_query = re.sub(r'<think>.*?</think>', '', sql_query, flags=re.DOTALL).strip()

            try:
                sqls = json.loads(sql_query)
            except Exception as e:
                raise

            sql_list.append(sqls)

        return sql_list

    def generate_datasource_json(self, user_question: str) -> Dict[str, Any]:
        """
        Generates a datasource JSON object based on the user question.

        Args:
            user_question (str): The user's question or request for report generation.

        Returns:
            Dict[str, Any]: A dictionary representing the datasource JSON object.

        Raises:
            JSONDecodeError: If the LLM response cannot be parsed as valid JSON
            Exception: If any other error occurs during the execution
        """
        max_retries = 0  # 额外重试次数
        attempt = 0
        while attempt <= max_retries:
            try:
                chain = self.query_filters_prompt_template | self.model
                response = chain.invoke({"user_question": user_question})

                json_string = response.content.strip()
                json_string = re.sub(r'<think>.*?</think>', '', json_string, flags=re.DOTALL).strip()

                if json_string.startswith("```json"):
                    json_string = json_string[7:]
                if json_string.endswith("```"):
                    json_string = json_string[:-3]
                return json.loads(json_string)
            except Exception as e:
                print(f"Error: Chain execution failed. Error: {e}")
                raise

    def get_slide_filters_json(self, template_slide: Dict):
        slide_filters = []
        slide_params = self.process_slide_params(template_slide)
        for slide_param in slide_params:
            item = self.generate_slide_filters_json(slide_param)
            slide_filters.append(item)
        return slide_filters

    def generate_slide_filters_json(self, slide_param: Dict) -> Dict[str, Any]:
        """
        Generates slide filters JSON from slide parameters with error handling and JSON cleanup.
        """
        try:
            chain = self.slide_filters_prompt_template | self.model
            response = chain.invoke({"slide_params": slide_param})

            json_string = response.content.strip()
            json_string = re.sub(r'<think>.*?</think>', '', json_string, flags=re.DOTALL).strip()

            if json_string.startswith("```json"):
                json_string = json_string[7:]
            if json_string.endswith("```"):
                json_string = json_string[:-3]

            return json.loads(json_string)

        except Exception as e:
            print(f"Error: Chain execution failed. Error: {e}")
            raise

    def _nearest_point(self, point, points):
        """
        Finds the nearest point from a list of points to the given point.

        Args:
            point: Target point as (x, y) tuple
            points: List of points as [(x1, y1), (x2, y2), ...]

        Returns:
            int: Index of the nearest point in the list
        """
        px, py = point
        best_dist = float('inf')
        best_idx = -1

        for i, (x, y) in enumerate(points):
            d = hypot(x - px, y - py)
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def process_slide_params(self, template_slide: Dict[str, Any]):
        """
        Processes slide parameters by extracting caption and table/chart elements,
        matching them based on spatial proximity, and converting data to DataFrame format.
        Returns a list of dictionaries containing caption, row headers, and column headers.
        """
        elements = template_slide.get("elements", [])
        caption_temps = []
        table_temp = []
        points = []
        pairs = []
        for element in elements:
            if element.get("role") in {'caption'}:
                caption_temps.append(element)
            if element.get("role") in {'table', 'chart-bar', 'chart-line'}:
                table_temp.append(element)
                points.append((element.get("layout").get('x'), element.get("layout").get('y')))

        for item in caption_temps:
            item_point = (item.get("layout").get('x'), item.get("layout").get('y'))
            nearest_point_idx = self._nearest_point(item_point, points)
            pairs.append((item, table_temp[nearest_point_idx]))

        slide_params = []
        for pair in pairs:
            data = pair[1].get('data')
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
                            df = pd.DataFrame(data)  # fallback
                else:
                    raise TypeError(f"Unsupported data type: {type(data)}")
            else:
                df = data
            if df.shape[1] < 1:
                raise ValueError("Expected at least 1 column of data")

            second_col_name = df.columns[0]
            df2 = df.set_index(second_col_name)

            column_headers = list(df2.columns)

            row_headers = list(df2.index)
            if df.shape[1] < 1:
                raise ValueError("Expected at least 1 column of data")

            dic = {
                'caption': pair[0].get("text"),
                'row_headers': row_headers,
                'column_headers': column_headers,
            }
            slide_params.append(dic)

        return slide_params

    def process_update_filters(self, query_filters: Dict[str, Any], slide_filters: List):
        city = query_filters.get('city').lower()
        update_filters = []
        try:
            for i, item in enumerate(slide_filters):
                dic = {
                    "connection": deepcopy(item.get("connection")),
                    "select_columns": deepcopy(item.get("select_columns")),
                    "filters": deepcopy(query_filters),
                    "fun_tool": {
                        "quadruples": deepcopy(item.get("fun_tool").get("quadruples")),
                        "args": {
                            "area_range_size": deepcopy(query_filters.get("area_range_size")),
                            "price_range_size": deepcopy(query_filters.get("price_range_size"))
                        }
                    }
                }
                table_list = item.get('connection').get('table')
                if table_list and '_new_house' in table_list:
                    dic['connection']['table'] = f"{city}_new_house"
                elif table_list and '_resale_house' in table_list:
                    dic['connection']['table']  = f"{city}_resale_house"

                for k, v in dic.get("filters").items():
                    if v == "default" and k in item.get("filters"):
                        dic["filters"][k] = item.get("filters").get(k)
                update_filters.append(dic)

        except Exception as e:
            return update_filters

        return update_filters


