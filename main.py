
import time
from tqdm import tqdm
from sql_generator import SqlGenerator
from tools_selector import ToolSelector
from yaml_processor import YamlProcessor
from database_manager import DatabaseManager
from conclusion_generator import ConclusionGenerator
from file_utils import find_target_csv_files, read_report_tasks_from_csv

def main():
    """

    """
    start_time = time.time()

    csv_files = find_target_csv_files(base_pattern="ReSlide/test/ReSlide_0*/*/template-*/temp/filename_to_label.csv")
    if not csv_files:
        print("No 'filename_to_label.csv' files found. Program exits.")
        return
    print(f"Found {len(csv_files)} CSV configuration files.")

    sql_generator = SqlGenerator()
    tool_selector = ToolSelector()
    conclusion_generator = ConclusionGenerator()
    database_manager = DatabaseManager()

    for csv_path in csv_files:
        print(f"\n{'=' * 20} Processing CSV file: {csv_path.parent}/{csv_path.name} {'=' * 20}")
        tasks = read_report_tasks_from_csv(csv_path)
        if not tasks:
            print("No valid tasks found in this CSV file.")
            continue
        print(f"Found {len(tasks)} tasks in this file.")

        for task in tqdm(tasks, desc="Processing Tasks", unit="task"):
            task_start_time = time.time()
            processor = YamlProcessor(task, sql_generator, database_manager, tool_selector, conclusion_generator)
            generated_data = processor.process_and_generate(task)
            processor.save_to_file(generated_data)
            task_end_time = time.time()
            task_elapsed_time = task_end_time - task_start_time
            print(f"Execution time for single task: {task_elapsed_time:.2f} seconds")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Program execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()