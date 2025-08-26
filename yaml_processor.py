# yaml_processor.py
from datetime import datetime

import yaml
import copy
from pathlib import Path
from typing import Dict, Any

from conclusion_generator import ConclusionGenerator
from database_manager import DatabaseManager
# 从其他模块导入依赖
from file_utils import ReportTask, load_yaml_file
from pptx_parser import PptxParser
from sql_generator import SqlGenerator
from tools_selector import ToolSelector


class YamlProcessor:
    """
    处理单个报告任务，整合信息并生成新的YAML文件。
    """
    def __init__(self, task: ReportTask, sql_generator: SqlGenerator, database_manager: DatabaseManager, tool_selector: ToolSelector, conclusion_generator: ConclusionGenerator):
        self.task = task
        self.sql_generator = sql_generator
        self.database_manager = database_manager
        self.tool_selector = tool_selector
        self.conclusion_generator = conclusion_generator
        self.pptx_parser = PptxParser(self.task.pptx_template_path)

    def _generate_output_slide(self, ground_truth_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据新数据生成 output_slide 部分。
        目前仅注入SQL查询，为未来功能留出接口。
        """
        # 深拷贝以避免修改原始数据
        output_slide = copy.deepcopy(ground_truth_data.get('output_slide', {}))
        
        # 为所有 content_elements 生成并注入SQL查询
        if 'content_elements' in output_slide and isinstance(output_slide['content_elements'], list):
            print(f"正在为 '{self.task.query[:30]}...' 生成SQL...")
            sql_query = self.sql_generator.generate_sql(self.task.query)
            print(f"  -> 生成的SQL: {sql_query}")

            for element in output_slide['content_elements']:
                element['sql_query'] = sql_query
                # 这里可以为未来的数据迁移和图表生成逻辑留出接口
                # 例如: element['data'] = execute_sql_and_get_data(sql_query)
                #       element['chart_image'] = generate_chart(element['data'])
        
        return output_slide

    def create_timestamped_folder(self) -> Path:
        """
        根据当前时间创建目录并返回路径（Path 对象）。
        - base: 根目录
        - fmt: 时间格式（strftime）
        """
        base = Path("data")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = base / stamp
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()

    def process_and_generate(self) -> Dict[str, Any]:
        """
        执行完整的处理流程，生成最终的YAML数据字典。

        Returns:
            一个包含所有生成信息的字典。
        """
        print("1. 解析用户意图生成 'data_source'...")
        new_data_source = self.sql_generator.generate_datasource_json(self.task.query)
        print(f"  -> 生成的数据源: {new_data_source}")

        print(f"2. 解析PPT模板意图: {self.task.pptx_template_path.name}")
        parsed_template_structure = self.pptx_parser.parse_slide(slide_idx=0)

        print("3. 根据用户需求与ppt解析生成SQL查询语句,接着检索数据存储到data目录  ...")
        '''sql_query的数据类型是list'''
        sql_query = self.sql_generator.generate_sql(user_question=self.task.query, slide_params=parsed_template_structure)
        print(f"  -> 生成的SQL: {sql_query}")
        data_path = self.create_timestamped_folder()
        try:
            self.database_manager.execute_query_save_data(sql_query, data_path)
        except Exception as e:
            print(f"  -> 错误: {e}")

        print("4. 给定用户需求与ppt意图自动调用工具  ...")
        try:
            tool_call_params = self.tool_selector.select_function_by_intent(data_source=new_data_source, slide_params=parsed_template_structure, data_path=data_path)
        except Exception as e:
            print(f"  -> 错误: {e}")

        print("5. 根据数据生成结论部分...")
        conclusion = self.conclusion_generator.get_conclusion(slide_params=parsed_template_structure, data_source=new_data_source, data_path=data_path)
        print(f"  -> 生成的结论是: {conclusion}")

        # 6. 组装最终的测评YAML结构
        eval_yaml = {
            'sql_query': sql_query,
            'tool_call_params': tool_call_params,
            'conclusion': conclusion,
        }

        #
        # # 4. 组装最终的YAML结构
        # generated_yaml = {
        #     'query': self.task.query,
        #     'data_source': new_data_source,  # 使用LLM生成的数据源
        #     'template_slide': parsed_template_structure.get('template_slide'), # 使用解析出的结构
        #     'output_slide': new_output_slide
        # }
        #
        return eval_yaml

    def save_to_file(self, data: Dict[str, Any]):
        """将生成的字典保存到YAML文件。"""
        # 在原始YAML文件同目录下生成新文件
        output_dir = self.task.ground_truth_yaml_path.parent
        output_filename = f"{self.task.ground_truth_yaml_path.stem}_generated.yaml"
        output_path = output_dir / output_filename

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(
                data, 
                f, 
                default_flow_style=False, 
                allow_unicode=True, 
                indent=2,
                sort_keys=False
            )
        print(f"✅ 成功生成YAML文件: {output_path}\n")

