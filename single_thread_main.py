# main.py

import os
from dotenv import load_dotenv

from conclusion_generator import ConclusionGenerator
from database_manager import DatabaseManager
# 从自定义模块中导入
from file_utils import find_target_csv_files, read_report_tasks_from_csv
from sql_generator import SqlGenerator
from tools_selector import ToolSelector
from yaml_processor import YamlProcessor

def main():
    """
    主函数：查找并批量处理所有报告生成任务。
    """
    # 从 .env 文件加载环境变量
    load_dotenv()
    base_url = os.getenv("BASE_URL")
    api_key = os.getenv("API_KEY")
    model_name = os.getenv("MODEL_NAME")

    if not api_key and not base_url:
        print("错误: 未找到 LLM 环境变量。")
        print("请在项目根目录下创建一个 .env 文件，并添加 BASE_URL='your_url' API_KEY='your_key'。")
        return

    print("开始批量生成报告YAML配置...")

    # 1. 查找所有包含任务的CSV文件
    csv_files = find_target_csv_files()
    if not csv_files:
        print("未找到任何 'filename_to_label.csv' 文件。程序退出。")
        return
    print(f"找到 {len(csv_files)} 个CSV配置文件。")

    # 2. 初始化SQL生成器 (只需一次)
    try:
        sql_generator = SqlGenerator(base_url=base_url, api_key=api_key, model_name=model_name)
        tool_selector = ToolSelector(base_url=base_url, api_key=api_key, model_name=model_name)
        conclusion_generator = ConclusionGenerator(base_url=base_url, api_key=api_key, model_name=model_name)
        database_manager = DatabaseManager()

    except ValueError as e:
        print(f"错误: 初始化SQL生成器失败: {e}")
        return

    success_count = 0
    error_count = 0

    # 3. 遍历所有CSV文件并处理其中的任务
    for csv_path in csv_files:
        print(f"\n{'='*20} 正在处理CSV文件: {csv_path.parent}/{csv_path.name} {'='*20}")
        tasks = read_report_tasks_from_csv(csv_path)

        if not tasks:
            print("此CSV文件中没有找到有效的任务。")
            continue

        print(f"在此文件中找到 {len(tasks)} 个任务。")

        for task in tasks:
            try:
                # 为每个任务创建一个处理器实例
                processor = YamlProcessor(task, sql_generator, database_manager, tool_selector, conclusion_generator)

                # 执行处理并生成最终的YAML数据
                generated_data = processor.process_and_generate()

                # 保存到文件
                processor.save_to_file(generated_data)

                success_count += 1
            except Exception as e:
                print(f"❌ 处理任务时发生严重错误: {task.query[:50]}... | 错误: {e}")
                error_count += 1
                import traceback
                traceback.print_exc()
    #
    # print(f"\n{'='*25} 批量处理完成！ {'='*25}")
    # print(f"✅ 成功生成: {success_count} 个YAML文件")
    # print(f"❌ 失败: {error_count} 个任务")

if __name__ == "__main__":
    main()


