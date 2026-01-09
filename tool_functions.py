import json
import re
from typing import List
import numpy as np
import pandas as pd
from tool_utils import *

def execute_analysis(options: str, input_path: str, output_path: str, price_range_size=1, area_range_size=20):
    """
    Executes data analysis operations based on specified options and saves results to output path.

    Args:
        options (str): JSON string containing analysis configuration including table type,
                      column names, database column mappings, and aggregation functions
        input_path (str): Path to the input data file to be processed
        output_path (str): Path where the processed results will be saved as Excel file
        price_range_size (int, optional): Size increment for price range binning, defaults to 1
        area_range_size (int, optional): Size increment for area range binning, defaults to 20

    Returns:
        str: JSON string containing status, stop flag, and output path
    """
    options = json.loads(options)
    df_data = load_data(input_path)
    df_data_copy = df_data.copy()
    table_type = options[0]
    table_col_names = options[1][0]
    database_col_names = options[2]
    arg_funs = options[3]
    option_range =[]
    if table_type == 'field-constraint' or table_type == 'constraint-field':
        option_range.append(options[1][1][0])
        table_args = options[1][1]
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

        if  table_type == 'constraint-field':
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

    export_to_excel(result, output_path)

    return json.dumps({
        "status": "success",
        "stop": True,
        "output_path": output_path
    }, ensure_ascii=False)
