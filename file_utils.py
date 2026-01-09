import csv
import glob
import os
import yaml
from pathlib import Path
from typing import List, Dict, Any, NamedTuple

class ReportTask(NamedTuple):
    """
    Data structure for storing a single report task read from CSV.
    """
    pptx_template_path: Path
    query: str
    ground_truth_yaml_path: Path

def find_target_csv_files(base_pattern: str = "ReSlide/ReSlide_*/*/template-*/temp/filename_to_label.csv") -> List[Path]:
    """
    Find all target CSV files according to the specified pattern.

    Args:
        base_pattern: Glob pattern used to search for files.

    Returns:
        A list of Path objects containing all matched file paths.
    """
    return [Path(p) for p in glob.glob(base_pattern)]

def read_report_tasks_from_csv(csv_path: Path) -> List[ReportTask]:
    """
    Read all report tasks from the specified three-column CSV file.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        A list of ReportTask objects.
    """
    tasks = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for i, row in enumerate(reader, 0):
                if len(row) < 3:
                    continue
                pptx_path_str, query, yaml_path_str = row
                pptx_path = Path(pptx_path_str).resolve()
                yaml_path = Path(yaml_path_str).resolve()

                if not pptx_path.exists():
                    print(f"Warning: PPTX template path does not exist at row {i}: {pptx_path}, skipped.")
                    continue

                if not yaml_path.exists():
                    print(f"Warning: YAML path does not exist at row {i}: {yaml_path}, skipped.")
                    continue

                tasks.append(ReportTask(
                    pptx_template_path=pptx_path,
                    query=query.strip(),
                    ground_truth_yaml_path=yaml_path
                ))
    except Exception as e:
        print(f"Error: Failed to read CSV file {csv_path}: {e}")
    
    return tasks

def load_yaml_file(yaml_path: Path) -> Dict[str, Any]:
    """
    Safely load a YAML file.

    Args:
        yaml_path: Path to the YAML file to load.

    Returns:
        Dictionary containing the loaded YAML data.
    """
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error: Failed to load YAML file {yaml_path}: {e}")
        raise


def load_prompt_from_file(file_name: str) -> str:
    """
    Load prompt content from file.

    Args:
        file_name: Prompt file name.

    Returns:
        str: Content of the prompt.
    """
    possible_paths = [
        f"prompts/{file_name}"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as file:
                return file.read().strip()

    raise FileNotFoundError(f"Prompt file not found: {file_name}")


