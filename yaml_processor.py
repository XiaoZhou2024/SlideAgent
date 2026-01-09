from datetime import datetime
import yaml
import copy
from pathlib import Path
from typing import Dict, Any, Callable
from conclusion_generator import ConclusionGenerator
from database_manager import DatabaseManager
from file_utils import ReportTask, load_yaml_file
from pptx_parser2 import PptxParser
from sql_generator import SqlGenerator
from tools_selector import ToolSelector
from tool_functions import *



class YamlProcessor:
    def __init__(self, task: ReportTask, sql_generator: SqlGenerator, database_manager: DatabaseManager, tool_selector: ToolSelector, conclusion_generator: ConclusionGenerator):
        self.task = task
        self.sql_generator = sql_generator
        self.database_manager = database_manager
        self.tool_selector = tool_selector
        self.conclusion_generator = conclusion_generator
        self.pptx_parser = PptxParser(self.task.pptx_template_path)

    def _generate_output_slide(self, ground_truth_data: Dict[str, Any]) -> Dict[str, Any]:
        output_slide = copy.deepcopy(ground_truth_data.get('output_slide', {}))

        if 'content_elements' in output_slide and isinstance(output_slide['content_elements'], list):
            sql_query = self.sql_generator.generate_sql(self.task.query)

            for element in output_slide['content_elements']:
                element['sql_query'] = sql_query
        
        return output_slide

    def create_timestamped_folder(self) -> Path:
        base = Path("data")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = base / stamp
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()

    def create_timestamped_folder1(self) -> Path:

        base = Path("data")
        stamp = datetime.now().strftime("%m%d_%H%M%S")
        path = base / stamp
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()

    def load_yaml_data(self, yaml_path: Path):
        slide = self.pptx_parser.presentation.slides[0]
        data_list = self.pptx_parser._extract_pptx_elements1(slide)

        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            template_slide = data.get('template_slide', {})
            num = 0
            for i,element in enumerate(template_slide['elements']):
                if element.get('type') == 'chart' or element.get('type') == 'table':
                    template_slide['elements'][i]['data'] = data_list[num]
                    num+=1


            return template_slide

    def parse_ppt(self):
        try:
            query_filters = self.sql_generator.generate_datasource_json(self.task.query)


            parsed_template_structure = self.load_yaml_data(self.task.ground_truth_yaml_path)
            # parsed_template_structure = self.pptx_parser._match_caption_and_table(parsed_template_structure)
            # print("parsed_template_structure:",parsed_template_structure)


            slide_filters = self.sql_generator.get_slide_filters_json(parsed_template_structure)
            slide_filters = self.pptx_parser._match_caption_and_table1(parsed_template_structure,slide_filters)

            update_filters = self.sql_generator.process_update_filters(query_filters, slide_filters)


            return query_filters,  slide_filters, update_filters, parsed_template_structure
        except Exception as e:
            return None, None, None, None

    def parse_ppt_and_requirements_params(self):
        try:
            query_filters = self.sql_generator.generate_datasource_json(self.task.query)

            parsed_template_structure = self.pptx_parser.parse_slide_vlm(slide_idx=0)

            slide_filters = self.sql_generator.get_slide_filters_json(parsed_template_structure)

            update_filters = self.sql_generator.process_update_filters(query_filters, slide_filters)

            return query_filters, parsed_template_structure,slide_filters, update_filters
        except Exception as e:
            return None, None, None, None


    def generate_sql(self, update_filters: List):
        data_path = self.create_timestamped_folder()
        max_retries = 0
        attempt = 0
        while attempt <= max_retries:
            try:
                sql_query = self.sql_generator.generate_sql(update_filters)

                self.database_manager.execute_query_save_data(sql_query, data_path)
                return sql_query, data_path
            except Exception as e:
                print(f"{e}")
                return ['sql_error'], data_path


    def get_standard_answer_sql(self, task: ReportTask):
        with open(task.ground_truth_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        elements = data.get('output_slide', {}).get('content_elements', [])
        sql_list = []
        for element in elements:
            sql_list.append(element['sql_query'])
        sql_query = sql_list
        print(f"  -> : {sql_query}")
        data_path = self.create_timestamped_folder()
        try:
            self.database_manager.execute_query_save_data(sql_query, data_path)
        except Exception as e:
            print(f"  -> : {e}")
        return sql_query, data_path

    def _count_csv_files(self, dir_path: str | Path) -> int:
        p = Path(dir_path)
        return sum(1 for _ in p.glob("*.csv"))
    def run_with_optional(
            self,
            func: Callable[..., Any],
            data_path: str,
            project: Any,
            area_range_size: Any,
            price_range_size: Any,
    ) -> Any:

        base_path = Path(data_path)
        retrieval_path = base_path / "retrieval"
        processed_path = base_path / "processed"
        processed_path.mkdir(parents=True, exist_ok=True)
        input_path = str(retrieval_path / "0.csv")
        output_path = str(processed_path / "0.xlsx")

        payload = {}

        if project != 'default':
            payload["project"] = project
        if area_range_size != 'default':
            payload["area_range_size"] = area_range_size
        if price_range_size != 'default':
            payload["price_range_size"] = price_range_size

        return func(input_path = input_path, output_path = output_path, **payload)

    def generate_tool_call_params(self, query_filters: Dict, update_filters: list, data_path: Path):

        try:
            update_filters = self.tool_selector.select_function_by_intent(query_filters=query_filters,update_filters=update_filters, data_path=data_path)
            return update_filters
        except Exception as e:
            print(f"  -> : {e}")
            return update_filters

    def generate_conclusion(self, query_filters: Dict, template_slides: Dict[str, Any], data_path: Path) -> str:
        try:
            output_slide = self.conclusion_generator.get_conclusion(query_filters=query_filters, template_slide =template_slides,
                                                                   data_path=data_path)
            return output_slide
        except Exception as e:
            print(f"  -> : {e}")
            return ''

    def process_and_generate(self, task: ReportTask) -> Dict[str, Any]:

        return self._process_full_workflow()

    # 以下是拆分出的子方法
    def _process_full_workflow(self) -> Dict[str, Any]:
        query_filters, slide_filters, update_filters, template_slides = self.parse_ppt()

        try:
            sql_queries, data_path = self.generate_sql(copy.deepcopy(update_filters))
            for i, query in enumerate(sql_queries):
                update_filters[i]['sql_query'] = copy.deepcopy(query)
        except Exception as e:
            print(f"  -> : {e}")
            return ' '
        print("update_filters", update_filters)

        update_filters = self.generate_tool_call_params(
            copy.deepcopy(query_filters),
            copy.deepcopy(update_filters),
            data_path
        )
        output_slide = self.generate_conclusion(
            copy.deepcopy(query_filters),
            copy.deepcopy(template_slides),
            data_path
        )

        return {
            'query_filters': query_filters,
            'slide_filters': slide_filters,
            'update_filters': update_filters,
            'template_slide': template_slides,
            'output_slide': output_slide
        }

    def save_to_file(self, data: Dict[str, Any]):
        output_dir = self.task.ground_truth_yaml_path.parent
        output_filename = f"{self.task.ground_truth_yaml_path.stem}_generated_120b.yaml"
        output_path = output_dir / output_filename

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(
                copy.deepcopy(data),
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                canonical=False,
                indent=2,
                width=float('inf'),
            )
        print(f"✅ Successfully generated YAML file: {output_path}\n")


