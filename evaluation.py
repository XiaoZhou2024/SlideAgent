# evaluation.py

import sqlite3
import yaml
import glob
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set

def find_yaml_pairs(base_pattern: str = "ReSlide_*/template-*/**/*_generated.yaml") -> List[Tuple[Path, Path]]:
    """
    查找所有生成的YAML文件及其对应的原始Ground Truth YAML文件。

    Args:
        base_pattern (str): 用于搜索生成文件的glob模式。

    Returns:
        List[Tuple[Path, Path]]: 一个元组列表，每个元组包含 (生成的YAML路径, 原始YAML路径)。
    """
    generated_files = [Path(p) for p in glob.glob(base_pattern, recursive=True)]
    pairs = []
    for gen_file in generated_files:
        original_file = Path(str(gen_file).replace("_generated.yaml", ".yaml"))
        if original_file.exists():
            pairs.append((gen_file, original_file))
        else:
            print(f"警告: 找不到与 {gen_file.name} 对应的原始YAML文件。")
    return pairs

def extract_sql_from_yaml(yaml_path: Path) -> str:
    """
    从YAML文件中提取第一条SQL查询语句。

    Args:
        yaml_path (Path): YAML文件的路径。

    Returns:
        str: 提取到的SQL查询语句，如果找不到则返回空字符串。
    """
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # SQL语句在 output_slide -> content_elements 列表中
        elements = data.get('output_slide', {}).get('content_elements', [])
        if elements and isinstance(elements, list) and 'sql_query' in elements[0]:
            return elements[0]['sql_query'].strip()
    except Exception as e:
        print(f"错误: 读取或解析YAML文件 {yaml_path} 时失败: {e}")
    return ""

def setup_database(conn: sqlite3.Connection):
    """
    在内存数据库中创建表并插入虚拟数据用于测试。
    """
    cursor = conn.cursor()
    # 1. 创建表
    cursor.execute('''
    CREATE TABLE new_house (
        supply_sets INTEGER,
        trade_sets INTEGER,
        city_name TEXT,
        district_name TEXT,
        block_name TEXT,
        date_code TEXT
    )
    ''')
    
    # 2. 插入多样化的虚拟数据
    dummy_data = [
        (10, 8, '北京市', '怀柔区', '怀柔区', '2021-05-10'),
        (12, 10, '北京市', '怀柔区', '怀柔区', '2022-06-15'),
        (5, 4, '北京市', '密云区', '密云区', '2020-08-20'),
        (20, 18, '深圳市', '龙岗区', '龙岗中心城', '2021-01-01'),
        (25, 22, '深圳市', '龙岗区', '龙岗中心城', '2022-11-30'),
        (15, 13, '广州市', '南沙区', '黄阁', '2023-03-12'),
        (100, 80, '北京市', '怀柔区', '怀柔区', '2023-12-30'), # 边界日期
    ]
    cursor.executemany('INSERT INTO new_house VALUES (?, ?, ?, ?, ?, ?)', dummy_data)
    conn.commit()
    print("内存数据库已创建并填充虚拟数据。")

def compare_sql_execution(conn: sqlite3.Connection, generated_sql: str, ground_truth_sql: str) -> bool:
    """
    执行两个SQL查询并比较它们的结果是否完全相同。

    Args:
        conn (sqlite3.Connection): 数据库连接。
        generated_sql (str): LLM生成的SQL。
        ground_truth_sql (str): 真实的SQL。

    Returns:
        bool: 如果结果相同则返回True，否则返回False。
    """
    if not generated_sql or not ground_truth_sql:
        return False # 如果任一SQL为空，则视为不匹配

    cursor = conn.cursor()
    
    try:
        # 执行生成的SQL
        cursor.execute(generated_sql)
        generated_results = set(cursor.fetchall())
    except sqlite3.Error as e:
        print(f"  -> 执行生成的SQL时出错: {e}")
        return False # SQL语法错误，视为不匹配

    try:
        # 执行真实的SQL
        cursor.execute(ground_truth_sql)
        truth_results = set(cursor.fetchall())
    except sqlite3.Error as e:
        print(f"  -> 执行真实的SQL时出错: {e}")
        return False # 真实SQL也可能存在问题

    # 比较结果集
    return generated_results == truth_results

def main():
    """
    主函数：执行完整的SQL评估流程。
    """
    print("开始进行SQL生成准确性评估...")
    
    # 1. 查找所有需要比较的YAML文件对
    yaml_pairs = find_yaml_pairs()
    if not yaml_pairs:
        print("未找到任何 `*_generated.yaml` 文件进行评估。")
        return

    print(f"找到 {len(yaml_pairs)} 对YAML文件进行比较。")
    
    # 2. 设置内存数据库
    conn = sqlite3.connect(":memory:")
    setup_database(conn)
    
    total_files = len(yaml_pairs)
    match_count = 0
    results_log = []

    # 3. 遍历并评估每一对文件
    for i, (gen_path, truth_path) in enumerate(yaml_pairs, 1):
        print(f"\n--- 正在评估第 {i}/{total_files} 对文件 ---")
        print(f"  生成文件: {gen_path.name}")
        print(f"  真实文件: {truth_path.name}")

        gen_sql = extract_sql_from_yaml(gen_path)
        truth_sql = extract_sql_from_yaml(truth_path)

        # **[FIX]** 清理SQL语句以兼容SQLite
        # 移除 "public." 前缀和末尾的分号
        gen_sql = gen_sql.replace("public.", "").rstrip(';')
        truth_sql = truth_sql.replace("public.", "").rstrip(';')
        
        is_match = compare_sql_execution(conn, gen_sql, truth_sql)
        
        if is_match:
            match_count += 1
            print("  -> 结果: ✅ 匹配 (Execution Match)")
        else:
            print("  -> 结果: ❌ 不匹配")
        
        results_log.append({
            "generated_file": gen_path.name,
            "ground_truth_file": truth_path.name,
            "generated_sql": gen_sql,
            "ground_truth_sql": truth_sql,
            "is_match": "Yes" if is_match else "No"
        })

    # 4. 关闭数据库连接
    conn.close()

    # 5. 打印最终评估报告
    accuracy = (match_count / total_files) * 100 if total_files > 0 else 0
    print("\n" + "="*30)
    print("      评估结果汇总")
    print("="*30)
    print(f"总共比较文件数: {total_files}")
    print(f"执行结果匹配数: {match_count}")
    print(f"执行准确率 (EX): {accuracy:.2f}%")
    print("="*30)

    # 6. 将详细结果保存到CSV文件
    output_csv_path = Path("evaluation_results.csv")
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["generated_file", "ground_truth_file", "generated_sql", "ground_truth_sql", "is_match"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_log)
    
    print(f"\n详细评估日志已保存到: {output_csv_path}")

if __name__ == "__main__":
    main()
