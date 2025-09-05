# main.py

import os
import concurrent.futures

import pythoncom
from dotenv import load_dotenv
from typing import Optional

from conclusion_generator import ConclusionGenerator
from database_manager import DatabaseManager
# 从自定义模块中导入
from file_utils import find_target_csv_files, read_report_tasks_from_csv, ReportTask
from sql_generator import SqlGenerator
from tools_selector import ToolSelector
from yaml_processor import YamlProcessor

def process_task_wrapper(task: ReportTask, sql_generator: SqlGenerator,database_manager: DatabaseManager, tool_selector: ToolSelector, conclusion_generator: ConclusionGenerator) -> bool:
    """
    封装单个任务处理逻辑的函数，用于并发执行。
    返回 True 表示成功，False 表示失败。
    """
    pythoncom.CoInitialize()
    try:
        # 为每个任务创建一个处理器实例
        processor = YamlProcessor(task, sql_generator, database_manager, tool_selector, conclusion_generator)

        # 执行处理并生成最终的YAML数据
        generated_data = processor.process_and_generate(task)

        # 保存到文件
        processor.save_to_file(generated_data)
        
        return True
    except Exception as e:
        print(f"❌ 处理任务时发生严重错误: {task.query[:50]}... | 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    主函数：查找并以并发方式批量处理所有报告生成任务。
    """
    print("开始批量生成报告YAML配置...")

    # 1. 查找所有包含任务的CSV文件
    csv_files = find_target_csv_files()
    print(csv_files)
    if not csv_files:
        print("未找到任何 'filename_to_label.csv' 文件。程序退出。")
        return
    print(f"找到 {len(csv_files)} 个CSV配置文件。")

    # 2. 初始化SQL生成器 (只需一次)
    try:
        sql_generator = SqlGenerator()
        tool_selector = ToolSelector()
        conclusion_generator = ConclusionGenerator()
        database_manager = DatabaseManager()

    except ValueError as e:
        print(f"错误: 初始化SQL生成器失败: {e}")
        return


    # 2. 从所有CSV文件中收集所有任务
    all_tasks = []
    for csv_path in csv_files:
        tasks_from_csv = read_report_tasks_from_csv(csv_path)
        if tasks_from_csv:
            print(f"从 {csv_path.name} 中加载了 {len(tasks_from_csv)} 个任务。")
            all_tasks.extend(tasks_from_csv)
    
    if not all_tasks:
        print("所有CSV文件中均未找到有效任务。程序退出。")
        return
    
    print(f"\n总共找到 {len(all_tasks)} 个任务，准备开始并发处理...")

    success_count = 0
    error_count = 0

    # 4. 使用线程池并发处理所有任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # 提交所有任务到线程池
        future_to_task = {executor.submit(process_task_wrapper, task, sql_generator, database_manager, tool_selector, conclusion_generator): task for task in all_tasks}
        
        # as_completed 会在任务完成时立即返回结果
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result_is_success = future.result()
                if result_is_success:
                    success_count += 1
                else:
                    error_count += 1
            except Exception as exc:
                print(f"任务 {task.query[:30]}... 生成了一个未捕获的异常: {exc}")
                error_count += 1
    
    print(f"\n{'='*25} 批量处理完成！ {'='*25}")
    print(f"✅ 成功生成: {success_count} 个YAML文件")
    print(f"❌ 失败: {error_count} 个任务")

if __name__ == "__main__":
    main()
