# file_utils.py

import csv
import glob
import yaml
from pathlib import Path
from typing import List, Dict, Any, NamedTuple

class ReportTask(NamedTuple):
    """
    用于存储从CSV读取的单个报告任务的数据结构。
    """
    pptx_template_path: Path
    query: str
    ground_truth_yaml_path: Path

def find_target_csv_files(base_pattern: str = "ReSlide/ReSlide_07/template-*/temp/filename_to_label.csv") -> List[Path]:
    """
    根据指定的模式查找所有目标CSV文件。

    Args:
        base_pattern: 用于搜索文件的glob模式。

    Returns:
        一个包含所有匹配文件路径的Path对象列表。
    """
    return [Path(p) for p in glob.glob(base_pattern)]

def read_report_tasks_from_csv(csv_path: Path) -> List[ReportTask]:
    """
    从指定的三列CSV文件中读取所有报告任务。

    Args:
        csv_path: CSV文件的路径。

    Returns:
        一个包含ReportTask对象的列表。
    """
    tasks = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            header = next(reader)  # 跳过标题行

            for i, row in enumerate(reader, 1):
                if len(row) < 3:
                    print(f"警告: CSV文件 {csv_path} 的第 {i} 行少于3列，已跳过。")
                    continue
                
                pptx_path_str, query, yaml_path_str = row

                # 解析路径并使其相对于CSV文件所在目录
                # 假设CSV中的路径是相对于项目根目录的
                pptx_path = Path(pptx_path_str).resolve()
                yaml_path = Path(yaml_path_str).resolve()

                if not pptx_path.exists():
                    print(f"警告: 第 {i} 行的PPTX模板路径不存在: {pptx_path}，已跳过。")
                    continue
                
                if not yaml_path.exists():
                    print(f"警告: 第 {i} 行的YAML路径不存在: {yaml_path}，已跳过。")
                    continue

                tasks.append(ReportTask(
                    pptx_template_path=pptx_path,
                    query=query.strip(),
                    ground_truth_yaml_path=yaml_path
                ))
    except Exception as e:
        print(f"错误: 读取CSV文件 {csv_path} 时出错: {e}")
    
    return tasks

def load_yaml_file(yaml_path: Path) -> Dict[str, Any]:
    """安全地加载一个YAML文件。"""
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"错误: 加载YAML文件 {yaml_path} 时失败: {e}")
        raise

