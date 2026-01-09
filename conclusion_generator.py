import re
from copy import deepcopy
from math import hypot
from pathlib import Path
from typing import Any, Dict
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config import config
from file_utils import load_prompt_from_file


class ConclusionGenerator:
    """
    A class that generates conclusions based on template data and new input data using LLM.
    """
    def __init__(self, temperature: float = 0):
        """
        Initialize the ConclusionGenerator with specified temperature.

        Args:
            temperature (float): Temperature parameter for the LLM (default: 0)
        """
        self.model = ChatOpenAI(
            base_url=config.BASE_URL,
            api_key=config.API_KEY,
            temperature=temperature,
            model=config.MODEL_NAME
        )
        self.conclusion_prompt_template = self._create_conclusion_prompt_template()
        self.new_caption_prompt_template = self._create_new_caption_prompt_template()

    def _create_conclusion_prompt_template(self) -> ChatPromptTemplate:
        system_prompt = load_prompt_from_file("conclusion_prompt.txt")
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", """template_data:
                        {template_caption}
                        {template_data}
                        template_conclusion:    
                        {template_conclusion}

                        data:
                        {data_caption}
                        {data}
                        conclusion:
                            """)
        ])

    def _create_new_caption_prompt_template(self) -> ChatPromptTemplate:
        system_prompt = load_prompt_from_file("caption_prompt.txt")
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", """
                        table_template_caption: {template_caption}
                        params:{params}
                        table_caption:            
            """)
        ])

    def _nearest_point(self, point, points):
        px, py = point
        best_dist = float('inf')
        best_idx = -1

        for i, (x, y) in enumerate(points):
            d = hypot(x - px, y - py)
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def get_conclusion(self, query_filters: Dict, template_slide: Dict[str, Any], data_path: Path):
        try:
            base_path = Path(data_path)
            processed_path = base_path / "processed"
            processed_path.mkdir(parents=True, exist_ok=True)
            updated_elements = []
            elements = template_slide.get('elements', [])
            elements_table = [item for item in elements if
                              item.get('role') == 'table' or item.get('role') == 'chart-bar' or item.get(
                                  'role') == 'chart-line']
            updated_conclusion = []
            for item in elements:
                if item.get('role') == 'slide-title' or item.get('role') == 'body-text':
                    updated_elements.append(deepcopy(item))
                if item.get('role') == 'caption':
                    params = {
                        'city': query_filters.get('city'),
                        'block': query_filters.get('block'),
                        'project': query_filters.get('project'),
                        'start_date': query_filters.get('start_date'),
                        'end_date': query_filters.get('end_date')
                    }
                    get_new_caption_chain = self.new_caption_prompt_template | self.model
                    try:
                        caption_content = get_new_caption_chain.invoke(
                            {"template_caption": item.get("text"), "params": params}).content

                        caption_content = re.sub(r'<think>.*?</think>', '', caption_content, flags=re.DOTALL).strip()

                    except Exception as e:
                        print(f"Error: Failed to get new table title: {e}")

                    caption_point = (item.get('layout').get('x'), item.get('layout').get('y'))
                    table_points = []
                    for elements_table_layout in elements_table:
                        table_points.append((elements_table_layout.get('layout').get('x'), elements_table_layout.get('layout').get('y')))
                    best_idx = self._nearest_point(caption_point, table_points)
                    template_slide_table_data = elements_table[best_idx]
                    out_slide_table_data = pd.read_excel(processed_path / "0.xlsx")
                    chain = self.conclusion_prompt_template | self.model
                    template_conclusion = [d["text"] for d in updated_elements if d['role'] == 'body-text']
                    try:
                        response = chain.invoke(
                            {"template_caption": item.get("text"), "template_data": template_slide_table_data.get("data"),
                             "template_conclusion": template_conclusion[0],
                             "data_caption": caption_content,
                             "data": out_slide_table_data
                             })
                    except Exception as e:
                        print(f"Error: Failed to get new table summary: {e}")

                    conclusion = response.content.replace('*', '')
                    conclusion = re.sub(r'<think>.*?</think>', '', conclusion, flags=re.DOTALL).strip()
                    updated_conclusion.append(conclusion)

                    item['text'] = caption_content
                    template_slide_table_data['data'] = deepcopy(out_slide_table_data.to_dict('records'))

                    updated_elements.append(deepcopy(item))
                    updated_elements.append(deepcopy(template_slide_table_data))
                    break

            updated_elements[1]['text'] = deepcopy(updated_conclusion[0])

            output_slide = {
                "slide_size": deepcopy(template_slide.get("slide_size")),
                "elements": updated_elements,
            }
            return output_slide
        except Exception as e:
            return ""



