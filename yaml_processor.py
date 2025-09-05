# yaml_processor.py
from datetime import datetime

import yaml
import copy
from pathlib import Path
from typing import Dict, Any, Callable, Optional, Mapping

from conclusion_generator import ConclusionGenerator
from database_manager import DatabaseManager
# 从其他模块导入依赖
from file_utils import ReportTask, load_yaml_file
from pptx_parser2 import PptxParser
# from pptx_parser import PptxParser
from sql_generator import SqlGenerator
from tools_selector import ToolSelector
from data_process_tool import (
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
)


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

    def parse_ppt_and_requirements_params(self):
        print("1. 解析用户意图生成 'query_filters'...")
        query_filters = self.sql_generator.generate_datasource_json(self.task.query)
        print(self.task.query)
        print(query_filters)
        print(f"2. 解析PPT模板意图: {self.task.pptx_template_path.name}")
        # parsed_template_structure = self.pptx_parser.parse_slide(slide_idx=0)
        parsed_template_structure = self.pptx_parser.parse_slide_vlm(slide_idx=0)
        print("-"*60)
        print(parsed_template_structure)
        return query_filters, parsed_template_structure

    def generate_sql(self, parsed_template_structure: Dict[str, Any]):
        print("3. 根据用户需求与ppt解析生成SQL查询语句,接着检索数据存储到data目录  ...")
        data_path = self.create_timestamped_folder()
        max_retries = 2  # 额外重试次数
        attempt = 0
        while attempt <= max_retries:
            try:
                '''sql_query的数据类型是list'''
                sql_query = self.sql_generator.generate_sql(user_question = self.task.query, slide_params = parsed_template_structure)
                self.database_manager.execute_query_save_data(sql_query, data_path)
                print(f"  -> 生成的SQL: {sql_query}")
                return sql_query, data_path
            except Exception as e:
                attempt += 1
                if attempt <= max_retries:
                    print(f"警告: 第 {attempt} 次调用失败，正在重试... 错误: {e}")
                else:
                    print(f"错误: 调用链执行失败（已重试 {max_retries} 次仍失败）。错误: {e}")
                    return '', data_path


    def get_standard_answer_sql(self, task: ReportTask):
        with open(task.ground_truth_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        elements = data.get('output_slide', {}).get('content_elements', [])
        sql_list = []
        for element in elements:
            sql_list.append(element['sql_query'])
        sql_query = sql_list
        print(f"  -> 生成的SQL: {sql_query}")
        data_path = self.create_timestamped_folder()
        try:
            self.database_manager.execute_query_save_data(sql_query, data_path)
        except Exception as e:
            print(f"  -> 错误: {e}")
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
        """
        根据传入的可选参数构造调用，只把存在的参数传给目标函数。
        defaults 用于补齐未提供的值（可选）。
        """
        base_path = Path(data_path)
        retrieval_path = base_path / "retrieval"
        processed_path = base_path / "processed"
        processed_path.mkdir(parents=True, exist_ok=True)
        input_path = str(retrieval_path / "0.csv")
        output_path = str(processed_path / "0.xlsx")

        payload = {}

        # 组装调用参数：仅传“存在的”参数
        if project != 'default':
            payload["project"] = project
        if area_range_size != 'default':
            payload["area_range_size"] = area_range_size
        if price_range_size != 'default':
            payload["price_range_size"] = price_range_size

        return func(input_path = input_path, output_path = output_path, **payload)

    def get_standard_answer_tools(self, task: ReportTask, data_path):
        ##这块代码后续优化逻辑
        tool_dic = {
            '供应与成交套数及占比': supply_and_sales_counts_and_share,
            '供应与成交趋势': analyze_supply_sales_trend,
            '供应与成交套数统计': get_supply_sales_counts_stats,
            '面积-总价交叉分析': compute_area_price_cross_stats,
            '面积段房源数量统计': compute_area_num_stats,
            '价格段房源数量统计': compute_price_num_stats,
            '二手房面积-总价交叉分析': compute_area_price_cross_stats,
            '二手房面积段房源数量统计': compute_area_num_stats,
            '二手房价格段房源数量统计': compute_price_num_stats,
            '商品住宅历年市场容量': compute_market_capacity,
            '商品住宅历年套数量': compute_annual_traded_units,
            '商品住宅历年面积量': compute_annual_traded_area,
            '二手房成交套数及均价统计': compute_resale_house_total_and_avg_price,
            '二手房成交套数分布': compute_resale_house_transaction_count_distribution,
            '二手房成交均价分布': compute_resale_house_avg_price_distribution,
            '小区房价走势': get_recent_transaction_trend,
        }
        with open(task.ground_truth_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        conclusion = data.get('output_slide', {}).get('analysis',{}).get('content')
        print("正确结论是：", conclusion)

        elements = data.get('output_slide', {}).get('content_elements', [])
        data_source = data.get('data_source', {})
        fun_tool_list = []
        for element in elements:
            fun_tool_list.append({
                'tool': element['fun_tool'],
                'project': data_source['project'],
                'area_range_size': data_source['area_range_size'],
                'price_range_size': data_source['price_range_size']
            })
            self.run_with_optional(tool_dic[element['fun_tool']], data_path, data_source['project'], data_source['area_range_size'], data_source['price_range_size'])
            break
        print(f"  -> fun_tool: {fun_tool_list}")

        return fun_tool_list

    def generate_tool_call_params(self, new_data_source: Dict, parsed_template_structure: Dict[str, Any], data_path: Path):
        print("4. 给定用户需求与ppt意图自动调用工具  ...")
        try:
            tool_call_params = self.tool_selector.select_function_by_intent(data_source=new_data_source, slide_params=parsed_template_structure, data_path=data_path)
            return tool_call_params
        except Exception as e:
            print(f"  -> 错误: {e}")
            return ''

    def generate_conclusion(self, new_data_source: Dict, parsed_template_structure: Dict[str, Any], data_path: Path) -> str:
        print("5. 根据数据生成结论部分...")
        try:
            conclusion = self.conclusion_generator.get_conclusion(slide_params=parsed_template_structure,
                                                                  data_source=new_data_source, data_path=data_path)
            print(f"  -> 生成的结论是: {conclusion}")
            return conclusion
        except Exception as e:
            print(f"  -> 错误: {e}")
            return ''


    def process_and_generate(self, task: ReportTask) -> Dict[str, Any]:
        """
        执行完整的处理流程，生成最终的YAML数据字典。

        Returns:
            一个包含所有生成信息的字典。
        """
        FLAG = 'Sql_only'

        if FLAG == 'Sql_only':
            try:
                query_filters, parsed_template_structure = self.parse_ppt_and_requirements_params()
            except Exception as e:
                print(f"  -> 错误: {e}")
                return {
                    'query_filters': '',
                    'parsed_template_structure': '',
                }

            # try:
            #     sql_query, data_path = self.generate_sql(parsed_template_structure)
            # except Exception as e:
            #     print(f"  -> 错误: {e}")
            #     return {
            #         'sql_query': ''
            #     }

            eval_yaml = {
                'sql_query': 'sql_query',
            }

        elif FLAG == 'Con_only':
            new_data_source, parsed_template_structure = self.parse_ppt_and_requirements_params()
            sql_query, data_path = self.generate_sql(parsed_template_structure)
            tool_call_params = self.generate_tool_call_params(new_data_source, parsed_template_structure, data_path)
            conclusion = self.generate_conclusion(new_data_source, parsed_template_structure, data_path)
        elif FLAG == 'test':
            try:
                new_data_source, parsed_template_structure = self.parse_ppt_and_requirements_params()
            except Exception as e:
                raise ValueError(f"Failed to parse PPTX: {e}")
            sql_query, data_path = self.get_standard_answer_sql(task)
            # tool_call_params = self.generate_tool_call_params(new_data_source, parsed_template_structure, data_path)
            fun_tool_list = self.get_standard_answer_tools(task, data_path)
            conclusion = self.generate_conclusion(new_data_source, parsed_template_structure, data_path)


        eval_yaml = {
            'query_filters': query_filters,
            'parsed_template_structure': parsed_template_structure,
        }
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

