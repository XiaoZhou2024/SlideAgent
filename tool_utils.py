import re
from typing import List

import numpy as np
import pandas as pd

def load_data(csv_data_path):
    """Load data from CSV and sanitize date formats."""
    df = pd.read_csv(csv_data_path)
    standardization_steps = {
        'date_code': lambda x: pd.to_datetime(x, errors='coerce'),
        'supply_sets': lambda x: pd.to_numeric(x, errors='coerce'),
        'trade_sets': lambda x: pd.to_numeric(x, errors='coerce'),
        'dim_area': lambda x: pd.to_numeric(x, errors='coerce'),
        'dim_unit_price': lambda x: pd.to_numeric(x, errors='coerce'),
    }
    # 遍历标准化步骤并应用
    for column, func in standardization_steps.items():
        if column in df.columns:
            df[column] = func(df[column])
    return df

def load_data_standardization(df_data):
    """Load data from CSV and sanitize date formats."""
    df = df_data.copy()
    standardization_steps = {
        'date_code': lambda x: pd.to_datetime(x, errors='coerce'),
        'supply_sets': lambda x: pd.to_numeric(x, errors='coerce'),
        'trade_sets': lambda x: pd.to_numeric(x, errors='coerce'),
        'dim_area': lambda x: pd.to_numeric(x, errors='coerce'),
        'dim_unit_price': lambda x: pd.to_numeric(x, errors='coerce'),
    }
    for column, func in standardization_steps.items():
        if column in df.columns:
            df[column] = func(df[column])
    return df
def execute_analysis1(df_data,  options: tuple, price_range_size=1, area_range_size=20):
    df_data_copy = df_data.copy()
    df_data_copy = load_data_standardization(df_data_copy)
    table_type = options[0]
    table_col_names = options[1][0]
    database_col_names = options[2]
    arg_funs = options[3]
    option_range =[]
    if table_type == 'field-constraint' or table_type == 'constraint-field':
        option_range.append(options[1][1][0])
    else:
        for i in range(len(options[1])):
            option_range.append(options[1][i][0])

    if option_range == ['month'] or option_range == ['year']:
        if option_range == ['month']:
            df_data_copy['month'] = df_data_copy['date_code'].dt.to_period('M')
        else:
            df_data_copy['year'] = df_data_copy['date_code'].dt.year
    else:
        if option_range == ['price_range']:
            args = '{}-{} M'
            df_data_copy = create_bins(df_data_copy, 'dim_price', price_range_size, args)
        elif option_range == ['area_range']:
            args = '{}-{}m²'
            df_data_copy = create_bins(df_data_copy, 'dim_area', area_range_size, args)

        else:
            args = '{}-{}m²'
            args2 = '{}-{} M'
            print("price_range_size:", price_range_size)
            print("area_range_size:", area_range_size)
            df_data_bin = create_bins(df_data_copy, 'dim_area', area_range_size, args)
            df_data_copy = create_bins(df_data_bin, 'dim_price', price_range_size, args2)


    res_list = []
    if table_type == 'field-constraint' or table_type == 'constraint-field':
        filtered_data = df_data_copy
        for table_col_name, database_col_name, arg_fun in zip(table_col_names, database_col_names, arg_funs):
            if database_col_name == ['dim_area', 'supply_sets']:
                database_col_name = 'dim_area'
                filtered_data = df_data_copy[df_data_copy['supply_sets'] == 1.0]
            elif database_col_name == ['dim_area', 'trade_sets']:
                database_col_name = 'dim_area'
                filtered_data = df_data_copy[df_data_copy['trade_sets'] == 1.0]

            res = aggregate_data(filtered_data,
                                 option_range,
                                 table_col_name,
                                 arg_fun,
                                 database_col_name)
            res_list.append(res)

        result = res_list[0]
        for res in res_list[1:]:
            result = pd.merge(result, res, on=option_range[0], how='outer')

        if option_range == ['area_range'] or option_range == ['price_range']:
            result = compact_table(result, keep_rows=15)

        if table_type == 'constraint-field':
            result = result.set_index('year').T
            result.columns.name = None
            result.reset_index(inplace=True)
            result.rename(columns={'index': 'year'}, inplace=True)

    elif table_type == 'cross-constraint':
        filtered_data = df_data_copy
        if "price_range" in option_range:
                result = pd.crosstab(
                    filtered_data[option_range[0]],
                    filtered_data[option_range[1]],
                    margins=True,
                    margins_name='total'
                )
                result = compact_merge_dataframe_ranges(result,14,16)
                result.index.name = 'price_area'
                result.reset_index(inplace=True)
        else:
            if len(table_col_names) ==1 :
                agg_dict = {
                    table_col_names[0]: (database_col_names[0], arg_funs[0])
                }

            else:
                agg_dict = {
                    table_col_names[0]: (database_col_names[0], arg_funs[0]),
                    table_col_names[1]: (database_col_names[1], arg_funs[1])
                }
            grouped = filtered_data.groupby(option_range, observed=True).agg(
                **agg_dict
            )
            df_data_pivot = grouped.reset_index()

            if len(table_col_names) == 1:
                df_data_pivot = df_data_pivot.pivot(
                    index=option_range[0],
                    columns=option_range[1],
                    values=[table_col_names[0]]
                )
            else:
                df_data_pivot = df_data_pivot.pivot(
                    index=option_range[0],
                    columns=option_range[1],
                    values=[table_col_names[0], table_col_names[1]]
                )

            df_data_pivot.columns = [
                f"{metric}({year})" for metric, year in df_data_pivot.columns
            ]
            for col in df_data_pivot.columns:
                if col != option_range[0]:
                    df_data_pivot[col] = df_data_pivot[col].fillna(0).astype(int)

            final_result = df_data_pivot.reset_index()
            result = compact_area_table_1(final_result)

    print("result：\n", result)
    return result

def export_to_excel(df: pd.DataFrame, output_path: str, sheet_name: str = "Sheet1"):
    with pd.ExcelWriter(output_path) as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        print("supply_and_sales_counts_and_share executed successfully, data has been saved to Excel file:", output_path)

def create_bins(df, column_name, range_size, table_args):
    """
    Create bins for specified column based on the given range size.
    """
    min_value = df[column_name].min()
    max_value = df[column_name].max()
    if column_name == 'dim_price':
        range_size = int(float(range_size)*100)
    range_size = int(range_size)

    start = int(min_value // range_size) * range_size
    end = int((max_value // range_size) + 1) * range_size
    bins = list(range(start, end + range_size, range_size))

    if column_name == 'dim_area':
        labels = [table_args.format(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]
        df['area_range'] = pd.cut(df['dim_area'], bins=bins, labels=labels, right=False, include_lowest=True)
        return df
    elif column_name == 'dim_price':
        labels = [table_args.format(round(bins[i]/100,2), round(bins[i + 1]/100,2)) for i in range(len(bins) - 1)]
        df['price_range'] = pd.cut(df['dim_price'], bins=bins, labels=labels, right=False, include_lowest=True)
        return df
    else:
        raise ValueError("bins_lables error")


def aggregate_data(df: pd.DataFrame, group_args: List[str], col_name: str, agg_func: str, *agg_args) -> pd.DataFrame:
    """
    Aggregates data based on specified columns and aggregation function.

    Args:
        df (pd.DataFrame): The dataframe to be processed.
        group_args (list): List of column names to group by.
        col_name (str): The column name for the aggregated result in the returned dataframe.
        agg_func (str): Name of the aggregation function, such as 'count' or 'sum'.
        *agg_args: Additional aggregation arguments depending on the aggregation function.

    Returns:
        pd.DataFrame: A dataframe containing the aggregated results.
    """
    agg_dict = {col_name: (agg_args[0] if agg_args else col_name, agg_func)}
    result=df.groupby(group_args, observed=False).agg(**agg_dict).reset_index()
    result[col_name] = result[col_name].astype(int)
    return result

def calculate_ratio(numerator: np.array, denominator: np.array) -> np.array:
    """
    Calculate ratio avoiding division by zero errors
    :param numerator: Numerator
    :param denominator: Denominator
    :return: Ratio array
    """
    return np.where(denominator > 0, (numerator / denominator).round(2), 0)


def calculate_percentage(part: np.array, whole: np.array) -> np.array:
    """
    Calculate percentage
    :param part: Part value
    :param whole: Whole value
    :return: Percentage array
    """
    return (part / whole).fillna(0).map(lambda x: '{:.1%}'.format(x))

def compact_table(df, keep_rows=15):
    detail_df = df.copy()
    col_list = detail_df.columns.tolist()
    range_col = col_list[0]
    def get_lower(area):
        nums = re.findall(r'\d+', str(area))
        return int(nums[0]) if nums else 0

    detail_df['lower'] = detail_df[range_col].apply(get_lower)
    detail_df = detail_df.sort_values('lower').reset_index(drop=True)

    if len(detail_df) <= keep_rows:
        result = detail_df.drop(columns='lower')
        return result

    keep_part = detail_df.iloc[:keep_rows]
    merge_part = detail_df.iloc[keep_rows:]

    merged_lower = merge_part['lower'].min()
    if range_col == 'area_range':
        merged_name = f'≥{merged_lower}㎡'
    else:
        merged_name = f'≥{merged_lower}万'

    if range_col=='area_range' or range_col=='price_range':
        if len(col_list) == 3:
            merged_row = {
                range_col: merged_name,
                col_list[1]: merge_part[col_list[1]].sum(),
                col_list[2]: merge_part[col_list[2]].sum()
            }
        else:
            merged_row = {
                range_col: merged_name,
                col_list[1]: merge_part[col_list[1]].sum(),
            }
    result = pd.concat([
        keep_part.drop(columns='lower'),
        pd.DataFrame([merged_row])
    ], ignore_index=True)
    print(result)
    return result

def compact_merge_dataframe_ranges(df, max_rows=10, max_cols=10):
    result_df = df.copy()
    def get_merge_label(range_str):
        if '.' in range_str:
            match = re.search(r'(\d+\.?\d*)-(\d+\.?\d*)([^\d]*)', str(range_str))
        else:
            match = re.search(r'(\d+)-(\d+)([^\d]*)', str(range_str))
        if match:
            end_val = match.group(2)
            unit = match.group(3)
            print("end_val:", end_val)
            print("unit:", unit)
            return f"≥{end_val}{unit}"
        return "≥other"

    summary_row = None
    summary_col = None

    if 'total' in result_df.index:
        summary_row = result_df.loc['total']
        result_df = result_df.drop('total')

    if 'total' in result_df.columns:
        summary_col = result_df['total']
        result_df = result_df.drop('total', axis=1)


    if len(result_df) > max_rows:
        kept_rows = result_df.iloc[:max_rows]
        merged_rows = result_df.iloc[max_rows:]

        merge_label = get_merge_label(kept_rows.index[-1])
        merged_data = merged_rows.sum()
        merged_data.name = merge_label

        result_df = pd.concat([kept_rows, merged_data.to_frame().T])

    if len(result_df.columns) > max_cols:
        kept_cols = result_df.columns[:max_cols]
        merged_cols = result_df.columns[max_cols:]

        merge_label = get_merge_label(kept_cols[-1])
        merged_data = result_df[merged_cols].sum(axis=1)

        result_df = result_df[kept_cols]
        result_df[merge_label] = merged_data

    if summary_col is not None:
        result_df['total'] = result_df.sum(axis=1)

    if summary_row is not None:
        result_df.loc['total'] = result_df.sum()

    return result_df

def compact_merge_price_or_area_ranges(df, max_rows=10):
    result_df = df.copy()
    label_col = None
    for col in ['area_range', 'price_range']:
        if col in result_df.columns:
            label_col = col
            break

    if len(result_df) > max_rows:
        kept_df = result_df.iloc[:max_rows]
        merged_df = result_df.iloc[max_rows:]
        merged_count = merged_df['count'].sum()

        if label_col:
            last_range = kept_df[label_col].iloc[-1]
        else:
            last_range = kept_df.index[-1]

        match = re.match(r'(\d+)-(\d+)([^\d]*)', str(last_range))
        if match:
            end_val = match.group(2)
            unit = match.group(3)
            merge_label = f"≥{end_val}{unit}"
        else:
            merge_label = "其他"

        merged_row = pd.DataFrame(
            {label_col: [merge_label], 'count': [merged_count]} if label_col else {'count': [merged_count]},
            index=[kept_df.index[-1]+1 if isinstance(kept_df.index[-1], int) else merge_label]
        )

        if label_col:
            result_df = pd.concat([kept_df, merged_row], ignore_index=True)
        else:
            result_df = pd.concat([kept_df, merged_row])

    return result_df


def compact_area_table_1(df, max_rows=16):

    df = df
    if len(df) <= max_rows:
        return df
    data = df.copy()

    data['lower_bound'] = data.iloc[:, 0].apply(lambda x: float(x.split('-')[0]))

    data = data.sort_values('lower_bound').reset_index(drop=True)
    keep_rows = data.iloc[:max_rows - 1].copy()
    merge_rows = data.iloc[max_rows - 1:].copy()
    merge_point = int(merge_rows['lower_bound'].min())
    merged_row = pd.Series(index=data.columns, dtype=object)
    merged_row.iloc[0] = f"≥{merge_point}㎡"
    for col in data.columns[1:-1]:
        merged_row[col] = pd.to_numeric(merge_rows[col], errors='coerce').sum()

    result = pd.concat([
        keep_rows.drop('lower_bound', axis=1),
        pd.DataFrame([merged_row.drop('lower_bound')])
    ], ignore_index=True)

    return result