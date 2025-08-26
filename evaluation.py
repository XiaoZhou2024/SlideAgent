# evaluation.py

import sqlite3

import pandas as pd
import yaml
import glob
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set

from pandas._testing import assert_frame_equal
from sqlalchemy import create_engine


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
def get_pg_engine(user, password, host, port, db):
    url = f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}'
    return create_engine(url)

def extract_sql_from_yaml(yaml_path: Path):
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
        data_source = data.get('data_source', {})
        conclusion = data.get('output_slide', {}).get('analysis', {}).get('content', '')
        sql_list = []
        tool_call_params = []
        for element in elements:
            sql_list.append(element['sql_query'])
            tool_call_params.append({'tool': element['fun_tool'], 'args': {'area_range_size': data_source['area_range_size'], 'price_range_size': data_source['price_range_size']}})
        return sql_list, tool_call_params, conclusion
    except Exception as e:
        print(f"错误: 读取或解析YAML文件 {yaml_path} 时失败: {e}")
    return ""

def extract_sql_from_gen_yaml(yaml_path: Path):
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
        return data['sql_query'], data['tool_call_params'], data['conclusion']

    except Exception as e:
        print(f"错误: 读取或解析YAML文件 {yaml_path} 时失败: {e}")
    return ""

def frames_equal(df1, df2) -> bool:
    try:
        # 列集合必须一致
        if set(df1.columns) != set(df2.columns):
            return False

        cols = sorted(df1.columns)
        df1_ = df1[cols].sort_values(by=cols).reset_index(drop=True)
        df2_ = df2[cols].sort_values(by=cols).reset_index(drop=True)

        pd.testing.assert_frame_equal(
            df1_, df2_,
            check_dtype=False,  # 忽略类型差异
            check_exact=False,  # 允许浮点误差
            rtol=1e-5, atol=1e-8
        )
        return True
    except AssertionError:
        return False

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

def compare_sql_execution(sql_engine, generated_sql: list, ground_truth_sql: list) -> bool:
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

    with sql_engine.connect() as conn:
        try:
            print(f"正在处理SQL：{generated_sql}")
            generated_results = pd.read_sql(generated_sql, conn)
        except Exception as e:
            print(f"  -> 执行生成的generated_sql时出错: {e}")
            return False  # SQL语法错误，视为不匹配
        try:
            print(f"正在处理SQL：{ground_truth_sql}")
            truth_results = pd.read_sql(ground_truth_sql, conn)
        except Exception as e:
            print(f"  -> 执行生成的ground_truth_sql时出错: {e}")
            return False  # SQL语法错误，视为不匹配


    # 比较结果集
    return frames_equal(generated_results, truth_results)

def compare_tools_select(generated_tool_select, ground_truth_tool_select):
    tool_dic = {
        "supply_and_sales_counts_and_share": '供应与成交套数及占比',
        "analyze_supply_sales_trend": '供应与成交趋势',
        "get_supply_sales_counts_stats": '供应与成交套数统计',
        "compute_area_price_matrix": '面积-总价交叉分析',
        "compute_area_num_stats": '面积段房源数量统计',
        "compute_price_num_stats": '价格段房源数量统计',

        "compute_market_capacity": '商品住宅历年市场容量',
        "compute_annual_traded_units": '商品住宅历年套数量',
        "compute_annual_traded_area": '商品住宅历年面积量',
        "compute_resale_house_total_and_avg_price": '二手房成交套数及均价统计',
        "compute_resale_house_transaction_count_distribution": '二手房成交套数分布',
        "compute_resale_house_avg_price_distribution": '二手房成交均价分布',
        "get_recent_transaction_trend": '小区房价走势',
    }
    print(generated_tool_select['tool'])
    print(ground_truth_tool_select['tool'])
    tool_equal = tool_dic[generated_tool_select['tool']] == ground_truth_tool_select['tool']
    area_equal = generated_tool_select['args']['area_range_size'] == ground_truth_tool_select['args']['area_range_size']
    price_equal = generated_tool_select['args']['price_range_size'] == ground_truth_tool_select['args']['price_range_size']

    return tool_equal and area_equal and price_equal

def compare_conclusions(generated_conclusions: str, ground_truth_conclusions: str) -> bool:
    generated_conclusions = generated_conclusions.replace(' ', '')
    ground_truth_conclusions = ground_truth_conclusions.replace(' ', '')
    print(f"  -> 生成的结论: {generated_conclusions}")
    print(f"  -> 真实结论: {ground_truth_conclusions}")
    return generated_conclusions == ground_truth_conclusions


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
    engine = get_pg_engine(
        user='ikun',
        password='wwwhelloworld111',
        host='frp.bnuzh.top',
        port='14532',
        db='RealEstate'
    )

    total_files = len(yaml_pairs)
    sql_match_count = 0
    tool_match_count = 0
    conclusions_match_count = 0
    results_log = []

    # 3. 遍历并评估每一对文件
    for i, (gen_path, truth_path) in enumerate(yaml_pairs, 1):
        print(f"\n--- 正在评估第 {i}/{total_files} 对文件 ---")
        print(f"  生成文件: {gen_path.name}")
        print(f"  真实文件: {truth_path.name}")

        gen_sql, gen_tool_call_params, gen_conclusion = extract_sql_from_gen_yaml(gen_path)
        truth_sql, truth_tool_call_params, truth_conclusion = extract_sql_from_yaml(truth_path)

        for i in range(len(gen_sql)):
            sql_is_match = compare_sql_execution(engine, gen_sql[i], truth_sql[i])
            print(truth_tool_call_params[i])
            tool_is_match = compare_tools_select(gen_tool_call_params[i], truth_tool_call_params[i])

            if sql_is_match:
                sql_match_count += 1
                print("  -> 结果: ✅ 匹配 (SQL Execution Match)")
            else:
                print("  -> 结果: ❌ 不匹配(SQL Execution Match)")

            if tool_is_match:
                tool_match_count += 1
                print("  -> 结果: ✅ 匹配 (Tool Execution Match)")
            else:
                print("  -> 结果: ❌ 不匹配(Tool Execution Match)")


        conclusions_is_match = compare_conclusions(gen_conclusion,truth_conclusion)
        if conclusions_is_match:
            conclusions_match_count += 1
            print("  -> 结果: ✅ 匹配 (Conclusion Execution Match)")
        else:
            print("  -> 结果: ❌ 不匹配(Conclusion Execution Match)")



    #     results_log.append({
    #         "generated_file": gen_path.name,
    #         "ground_truth_file": truth_path.name,
    #         "generated_sql": gen_sql,
    #         "ground_truth_sql": truth_sql,
    #         "is_match": "Yes" if is_match else "No"
    #     })
    #
    #
    #
    # # 5. 打印最终评估报告
    # accuracy = (match_count / total_files) * 100 if total_files > 0 else 0
    # print("\n" + "="*30)
    # print("      评估结果汇总")
    # print("="*30)
    # print(f"总共比较文件数: {total_files}")
    # print(f"执行结果匹配数: {match_count}")
    # print(f"执行准确率 (EX): {accuracy:.2f}%")
    # print("="*30)
    #
    # # 6. 将详细结果保存到CSV文件
    # output_csv_path = Path("evaluation_results.csv")
    # with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    #     fieldnames = ["generated_file", "ground_truth_file", "generated_sql", "ground_truth_sql", "is_match"]
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #     writer.writerows(results_log)
    #
    # print(f"\n详细评估日志已保存到: {output_csv_path}")

if __name__ == "__main__":
    main()
