import re
import numpy as np
import pandas as pd

def supply_and_sales_counts_and_share(input_path: str, output_path: str, area_range_size: int=20):
    """
    供应与成交套数及占比
    """
    data=pd.read_csv(input_path)
    block_df = data.copy()
    # 确定面积区间范围
    min_area = block_df['dim_area'].min()
    max_area = block_df['dim_area'].max()

    # 构建均匀分段区间
    start = int(min_area // area_range_size) * area_range_size
    end = int((max_area // area_range_size) + 1) * area_range_size
    area_bins = list(range(start, end + area_range_size, area_range_size))

    labels = [f"{area_bins[i]}-{area_bins[i+1]}㎡" for i in range(len(area_bins)-1)]
    block_df['面积段'] = pd.cut(block_df['dim_area'], bins=area_bins, labels=labels, right=False, include_lowest=True)

    # 按面积段分组统计
    result = block_df.groupby('面积段', observed=False).agg(
        供应套数=('supply_sets', 'count'),
        成交套数=('trade_sets', 'count')
    ).reset_index()

    # 计算供求比（避免除零问题）
    result['供求比'] = np.where(
        result['成交套数'] > 0,
        (result['供应套数'] / result['成交套数']).round(2),
        0
    )

    # 计算成交占比
    total_trade = result['成交套数'].sum()
    result['成交占比'] = (result['成交套数'] / total_trade).fillna(0).map(lambda x: '{:.1%}'.format(x))

    # 添加汇总行
    total_supply = result['供应套数'].sum()
    total_ratio = round(total_supply / total_trade if total_trade != 0 else 0, 1)

    total_row = pd.DataFrame([{
        '面积段': '总计',
        '供应套数': total_supply,
        '成交套数': total_trade,
        '供求比': total_ratio,
        '成交占比': '100%'  # 总成交占比为100%
    }])

    # 合并结果
    result = pd.concat([result, total_row], ignore_index=True)

    result = compact_area_table(result, keep_rows=15)
    with pd.ExcelWriter(output_path) as writer:
        result.to_excel(writer, index=False)


def analyze_supply_sales_trend(input_path: str, output_path: str, area_range_size: int=20):
    """
    供应与成交趋势
    """
    data=pd.read_csv(input_path)
    # 日期标准化
    block_df = data.copy()
    block_df['date_code'] = pd.to_datetime(block_df['date_code'], errors='coerce')  # 处理 2021/11/26 这种格式

    # 确定面积区间范围
    min_area = block_df['dim_area'].min()
    max_area = block_df['dim_area'].max()

    # 构建均匀分段区间
    start = int(min_area // area_range_size) * area_range_size
    end = int((max_area // area_range_size) + 1) * area_range_size
    area_bins = list(range(start, end + area_range_size, area_range_size))

    # 将 date_code 转换为 datetime 类型并提取年份
    block_df['date_code'] = pd.to_datetime(block_df['date_code'], errors='coerce')
    block_df['年份'] = block_df['date_code'].dt.year

    # 使用cut分段，并自定义分段labels
    labels = [f"{area_bins[i]}-{area_bins[i+1]}㎡" for i in range(len(area_bins)-1)]
    block_df['面积段'] = pd.cut(block_df['dim_area'], bins=area_bins, labels=labels, right=False, include_lowest=True)

    # 按「面积段」+「年份」分组统计
    grouped = block_df.groupby(['面积段', '年份'], observed=False).agg(
        供应套数=('supply_sets', 'count'),
        成交套数=('trade_sets', 'count')
    )
    # 重新组织MultiIndex
    pivot_df = grouped.unstack(level='年份')

    # 重新排列MultiIndex层次，使得年份在第一层，指标在第二层
    pivot_df = pivot_df.swaplevel(axis=1).sort_index(axis=1, level=0)

    # 添加总计行
    total_row = pivot_df.sum()
    total_df = pd.DataFrame([total_row], index=['总计'])
    final_df = pd.concat([pivot_df, total_df], axis=0)
    final_df.columns.names = [None, None]

    # ----------- 只要这2步关键代码即可展平成单行表头 -----------
    # 1. 重命名单层列名
    final_df.columns = [f"{year}{item}" for year, item in final_df.columns]
    # 2. 将面积段变为第一列
    final_df = final_df.reset_index()
    # ----------------------------------------------------------
    # 关键1：自动修正首列名为“面积段”
    if final_df.columns[0] == 'index':
        final_df.rename(columns={'index': '面积段'}, inplace=True)

    final_df = compact_area_table_1(final_df)
    with pd.ExcelWriter(output_path) as writer:
        final_df.to_excel(writer, index=False)
    # return final_df


def get_supply_sales_counts_stats(input_path: str, output_path: str, area_range_size: int=20):
    """
    供应与成交套数统计
    """
    block_dataframe = pd.read_csv(input_path)
    # 日期标准化
    block_df = block_dataframe.copy()

    # 确定面积区间范围
    min_area = block_df['dim_area'].min()
    max_area = block_df['dim_area'].max()

    # 构建均匀分段区间
    start = int(min_area // area_range_size) * area_range_size
    end = int((max_area // area_range_size) + 1) * area_range_size
    area_bins = list(range(start, end + area_range_size, area_range_size))

    # 使用cut分段，并自定义分段labels
    labels = [f"{area_bins[i]}-{area_bins[i+1]}㎡" for i in range(len(area_bins)-1)]
    block_df['面积段'] = pd.cut(block_df['dim_area'], bins=area_bins, labels=labels, right=False, include_lowest=True)

    # 按面积段分组统计
    result = block_df.groupby('面积段', observed=False).agg(
        供应套数=('supply_sets', 'count'),
        成交套数=('trade_sets', 'count')
    ).reset_index()

    final_result = result
    final_result = compact_area_table_2(final_result, keep_rows=15)

    with pd.ExcelWriter(output_path) as writer:
        final_result.to_excel(writer, index=False)
    # return final_df

def compute_area_price_cross_stats(input_path: str, output_path: str, area_range_size: int=20, price_range_size: int=100):
    """
    面积-总价交叉分析
    """

    block_dataframe = pd.read_csv(input_path)
    # 日期标准化
    block_df = block_dataframe.copy()

    # 确定面积区间范围
    min_area = block_df['dim_area'].min()
    max_area = block_df['dim_area'].max()

    print("min_area",min_area)
    print("max_area",max_area)
    # 构建均匀分段区间
    start = int(min_area // area_range_size) * area_range_size
    end = int((max_area // area_range_size) + 1) * area_range_size
    area_bins = list(range(start, end, area_range_size))

    # 确定总价区间范围
    min_price = block_df['dim_price'].min()
    max_price = block_df['dim_price'].max()

    # 构建均匀分段区间
    start = int(min_price // price_range_size) * price_range_size
    end = int((max_price // price_range_size) + 1) * price_range_size
    price_bins = list(range(start, end, price_range_size))

    # 添加最后一个区间上限
    area_bins.append(end)
    area_bins = sorted(set(area_bins))
    # print("area_bins:{}".format(area_bins))
    price_bins.append(end)
    # print("price_bins:{}".format(price_bins))
    price_bins = sorted(set(price_bins))
    # 创建面积区间标签
    area_labels = [f"{area}-{area + area_range_size}m²" for area in area_bins[:-1]]

    # 创建价格区间标签
    price_labels = [f"{price}-{price + price_range_size}万元" for price in price_bins[:-1]]

    # 将数据按照面积和价格分组
    block_df['area_range'] = pd.cut(block_df['dim_area'], bins=area_bins, labels=area_labels, right=False)
    block_df['price_range'] = pd.cut(block_df['dim_price'], bins=price_bins, labels=price_labels, right=False)

    # 创建交叉统计表
    cross_tab = pd.crosstab(
        block_df['area_range'],
        block_df['price_range'],
        margins=True,
        margins_name='汇总'
    )

    cross_tab = compact_merge_dataframe_ranges(cross_tab,14,16)

    with pd.ExcelWriter(output_path) as writer:
        cross_tab.to_excel(writer, index=False)

def compute_area_num_stats(input_path: str, output_path: str, area_range_size: int=20):
    """
    面积段房源数量统计
    """

    block_dataframe = pd.read_csv(input_path)
    # 日期标准化
    block_df = block_dataframe.copy()

    # 确定面积区间范围
    min_area = block_df['dim_area'].min()
    max_area = block_df['dim_area'].max()

    # 构建均匀分段区间
    start = int(min_area // area_range_size) * area_range_size
    end = int((max_area // area_range_size) + 1) * area_range_size
    area_bins = list(range(start, end, area_range_size))

    # 添加最后一个区间上限
    area_bins.append(end)

    # 创建面积区间标签
    area_labels = [f"{area}-{area + area_range_size}m²" for area in area_bins[:-1]]

    # 将数据按照面积分组
    block_df['area_range'] = pd.cut(block_df['dim_area'], bins=area_bins, labels=area_labels, right=False)

    # 统计每个面积段的房子数量
    area_count = block_df['area_range'].value_counts().sort_index()

    area_count = area_count.reset_index()
    area_count.columns = ['area_range', 'count']
    area_count = compact_merge_price_or_area_ranges(area_count)

    with pd.ExcelWriter(output_path) as writer:
        area_count.to_excel(writer, index=False)

def compute_price_num_stats(input_path: str, output_path: str, price_range_size: int=100):
    """
    价格段房源数量统计
    """

    block_dataframe = pd.read_csv(input_path)
    # 日期标准化
    block_df = block_dataframe.copy()


    # 确定总价区间范围
    min_price = block_df['dim_price'].min()
    max_price = block_df['dim_price'].max()

    # 构建均匀分段区间
    start = int(min_price // price_range_size) * price_range_size
    end = int((max_price // price_range_size) + 1) * price_range_size
    price_bins = list(range(start, end, price_range_size))

    # 添加最后一个区间上限
    price_bins.append(end)
    price_bins = sorted(set(price_bins))

    # 创建价格区间标签
    price_labels = [f"{price}-{price + price_range_size}万元" for price in price_bins[:-1]]


    # 将数据按照价格分组
    block_df['price_range'] = pd.cut(block_df['dim_price'], bins=price_bins, labels=price_labels, right=False)

    # 统计每个面积段的房子数量
    price_count = block_df['price_range'].value_counts().sort_index()
    price_count = price_count.reset_index()
    price_count.columns = ['price_range', 'count']
    price_count = compact_merge_price_or_area_ranges(price_count)
    with pd.ExcelWriter(output_path) as writer:
        price_count.to_excel(writer, index=False)

def compute_market_capacity(input_path: str, output_path: str):
    """
    商品住宅历年市场容量
    """

    block_df = pd.read_csv(input_path)
    block_df['date_code'] = pd.to_datetime(block_df['date_code'])
    block_df['year'] = block_df['date_code'].dt.year

    # 保证这些列数值化
    block_df['supply_sets'] = pd.to_numeric(block_df['supply_sets'], errors='coerce')
    block_df['trade_sets'] = pd.to_numeric(block_df['trade_sets'], errors='coerce')
    block_df['dim_area'] = pd.to_numeric(block_df['dim_area'], errors='coerce')

    # 供应面积与套数（供应面积：supply_sets==1的dim_area求和）
    supply_area = block_df[block_df['supply_sets'] == 1].groupby('year')['dim_area'].sum()
    supply_area = (supply_area/10000).round(2)
    supply_sets = block_df.groupby('year')['supply_sets'].sum()
    supply_sets = supply_sets.astype('int')

    # 成交面积与套数（成交面积：trade_sets==1的dim_area求和）
    trade_area = block_df[block_df['trade_sets'] == 1].groupby('year')['dim_area'].sum()
    trade_area = (trade_area/10000).round(2)
    trade_sets = block_df.groupby('year')['trade_sets'].sum()
    trade_sets = trade_sets.astype('int')

    trade_avg_price = block_df[block_df['trade_sets'] == 1].groupby('year')['dim_unit_price'].mean()
    trade_avg_price = trade_avg_price.round(2)

    # 合并
    result = pd.concat([
        supply_area.rename('供应面积（万m2）'),
        supply_sets.rename('供应套数'),
        trade_area.rename('成交面积（万m2）'),
        trade_sets.rename('成交套数'),
        trade_avg_price.rename('成交均价（元/m2）')
    ], axis=1).reset_index()

    # 按年份升序排列
    result = result.sort_values('year').reset_index(drop=True)

    # 替换所有NaN为0
    result = result.fillna(0)

    with pd.ExcelWriter(output_path) as writer:
        result.to_excel(writer, index=False)

def compute_annual_traded_units(input_path: str, output_path: str):
    """
    商品住宅历年套数量
    """
    block_df = pd.read_csv(input_path)
    block_df['date_code'] = pd.to_datetime(block_df['date_code'])
    block_df['year'] = block_df['date_code'].dt.year

    # 保证这些列数值化
    block_df['supply_sets'] = pd.to_numeric(block_df['supply_sets'], errors='coerce')
    block_df['trade_sets'] = pd.to_numeric(block_df['trade_sets'], errors='coerce')

    # 供应面积与套数（供应面积：supply_sets==1的dim_area求和）
    supply_sets = block_df.groupby('year')['supply_sets'].sum()
    supply_sets = supply_sets.astype('int')

    # 成交面积与套数（成交面积：trade_sets==1的dim_area求和）
    trade_sets = block_df.groupby('year')['trade_sets'].sum()
    trade_sets = trade_sets.astype('int')

    # 合并
    result = pd.concat([
        supply_sets.rename('供应套数'),
        trade_sets.rename('成交套数'),
    ], axis=1).reset_index()

    # 按年份升序排列
    result = result.sort_values('year').reset_index(drop=True)

    # 替换所有NaN为0
    result = result.fillna(0)

    # --- 拆分成两个 DataFrame ---
    sets_df = result[['year', '供应套数', '成交套数']]

    with pd.ExcelWriter(output_path) as writer:
        sets_df.to_excel(writer, index=False)

def compute_annual_traded_area(input_path: str, output_path: str):
    """
    商品住宅历年面积量
    """
    block_df = pd.read_csv(input_path)
    block_df['date_code'] = pd.to_datetime(block_df['date_code'])
    block_df['year'] = block_df['date_code'].dt.year

    # 保证这些列数值化
    block_df['supply_sets'] = pd.to_numeric(block_df['supply_sets'], errors='coerce')
    block_df['trade_sets'] = pd.to_numeric(block_df['trade_sets'], errors='coerce')
    block_df['dim_area'] = pd.to_numeric(block_df['dim_area'], errors='coerce')

    # 供应面积与套数（供应面积：supply_sets==1的dim_area求和）
    supply_area = block_df[block_df['supply_sets'] == 1].groupby('year')['dim_area'].sum()
    supply_area = (supply_area/10000).round(2)

    # 成交面积与套数（成交面积：trade_sets==1的dim_area求和）
    trade_area = block_df[block_df['trade_sets'] == 1].groupby('year')['dim_area'].sum()
    trade_area = (trade_area/10000).round(2)

    # 合并
    result = pd.concat([
        supply_area.rename('供应面积（万m2）'),
        trade_area.rename('成交面积（万m2）'),
    ], axis=1).reset_index()

    # 按年份升序排列
    result = result.sort_values('year').reset_index(drop=True)

    # 替换所有NaN为0
    result = result.fillna(0)

    # --- 拆分成两个 DataFrame ---
    area_df = result[['year', '供应面积（万m2）', '成交面积（万m2）']]

    with pd.ExcelWriter(output_path) as writer:
        area_df.to_excel(writer, index=False)

def compute_resale_house_total_and_avg_price(input_path: str, output_path: str):
    """
    二手房成交套数及均价统计
    """
    block_df = pd.read_csv(input_path)
    block_df['date_code'] = pd.to_datetime(block_df['date_code'])
    block_df['year'] = block_df['date_code'].dt.year

    # 保证这些列数值化
    block_df['trade_sets'] = pd.to_numeric(block_df['trade_sets'], errors='coerce')

    # 成交面积与套数（成交面积：trade_sets==1的dim_area求和）
    trade_area = block_df[block_df['trade_sets'] == 1].groupby('year')['dim_area'].sum()
    trade_area = (trade_area).round(2)
    trade_sets = block_df.groupby('year')['trade_sets'].sum()
    trade_sets = trade_sets.astype('int')

    ## 成交均价
    trade_price = block_df.groupby('year')['dim_unit_price'].mean()

    # 合并
    result = pd.concat([
        trade_area.rename('成交面积（m2）'),
        trade_sets.rename('成交套数'),
        trade_price.rename('成交均价（元/m2）')
    ], axis=1).reset_index()

    # 可选：填充缺失为0
    result_df = result.fillna(0).astype(int)

    with pd.ExcelWriter(output_path) as writer:
        result_df.to_excel(writer, index=False)



def compute_resale_house_transaction_count_distribution(input_path: str, output_path: str):
    """
    二手房成交套数分布
    """
    block_df = pd.read_csv(input_path)
    block_df['date_code'] = pd.to_datetime(block_df['date_code'])
    block_df['year'] = block_df['date_code'].dt.year

    # 保证这些列数值化
    block_df['trade_sets'] = pd.to_numeric(block_df['trade_sets'], errors='coerce')

    # 成交面积与套数（成交面积：trade_sets==1的dim_area求和）
    trade_area = block_df[block_df['trade_sets'] == 1].groupby('year')['dim_area'].sum()
    trade_area = (trade_area).round(2)
    trade_sets = block_df.groupby('year')['trade_sets'].sum()
    trade_sets = trade_sets.astype('int')

    ## 成交均价
    trade_price = block_df.groupby('year')['dim_unit_price'].mean()

    # 合并
    result = pd.concat([
        trade_sets.rename('成交套数'),
    ], axis=1).reset_index()

    # 可选：填充缺失为0
    result_df = result.fillna(0).astype(int)
    with pd.ExcelWriter(output_path) as writer:
        result_df.to_excel(writer, index=False)


def compute_resale_house_avg_price_distribution(input_path: str, output_path: str):
    """
    二手房成交均价分布
    """
    block_df = pd.read_csv(input_path)
    block_df['date_code'] = pd.to_datetime(block_df['date_code'])
    block_df['year'] = block_df['date_code'].dt.year

    # 保证这些列数值化
    block_df['trade_sets'] = pd.to_numeric(block_df['trade_sets'], errors='coerce')

    # 成交面积与套数（成交面积：trade_sets==1的dim_area求和）
    trade_area = block_df[block_df['trade_sets'] == 1].groupby('year')['dim_area'].sum()
    trade_area = (trade_area).round(2)
    trade_sets = block_df.groupby('year')['trade_sets'].sum()
    trade_sets = trade_sets.astype('int')

    ## 成交均价
    trade_price = block_df.groupby('year')['dim_unit_price'].mean()

    # 合并
    result = pd.concat([
        trade_price.rename('成交均价（元/m2）')
    ], axis=1).reset_index()

    # 可选：填充缺失为0
    result_df = result.fillna(0).astype(int)
    with pd.ExcelWriter(output_path) as writer:
        result_df.to_excel(writer, index=False)

def get_recent_transaction_trend(input_path: str, output_path: str,
                          project_name: str | None = None,
                          freq: str = 'M'   # 'M'：自然月；'Q'：季度
                         ):
    """
    小区房价走势
    """
    df = pd.read_csv(input_path)
    # 1. 日期标准化
    df = df.copy()
    df['date_code'] = pd.to_datetime(df['date_code'], errors='coerce')  # 处理 2021/11/26 这种格式

    # 2. 项目筛选
    if project_name is not None:
        df = df.loc[df['project_name'] == project_name]

    # 4. 计算平均单价
    # 先把 dim_unit_price 转成数值，防止有字符串
    df['dim_unit_price'] = pd.to_numeric(df['dim_unit_price'], errors='coerce')

    # 使用 PeriodIndex 聚合
    df['year_month'] = df['date_code'].dt.to_period(freq)
    avg_price = (
        df.groupby('year_month')['dim_unit_price']
          .mean()
          .sort_index()
    ).round(2)

    # 6. 将 PeriodIndex 变成字符串，好看
    avg_price.index = avg_price.index.astype(str)

    avg_price.name = 'avg_unit_price'  # 数值列的表头
    avg_price.index.name = 'date'  # 行索引的表头

    avg_price = avg_price.fillna(0)
    result_df = avg_price.to_frame().reset_index()

    with pd.ExcelWriter(output_path) as writer:
        result_df.to_excel(writer, index=False)


def compact_area_table(df, keep_rows=15):
    """
    当明细行数超过 keep_rows 时，把第 keep_rows+1 行及之后
    的数据合并成一行，面积段名称自动变成 '≥xxx㎡'。
    参数
    """
    total_label = '总计'
    # 0. 拆出汇总行（如果有）
    total_row = df[df['面积段'] == total_label]
    detail_df = df[df['面积段'] != total_label].copy()

    # 1. 取面积段下限，用于排序和后续判断
    def get_lower(area):
        nums = re.findall(r'\d+', str(area))
        return int(nums[0]) if nums else 0

    detail_df['lower'] = detail_df['面积段'].apply(get_lower)
    detail_df = detail_df.sort_values('lower').reset_index(drop=True)

    # 2. 判断是否需要合并
    if len(detail_df) <= keep_rows:
        result = detail_df.drop(columns='lower')
        return (pd.concat([result, total_row], ignore_index=True)
                if not total_row.empty else result)

    # 3. 需要合并：拆分保留区和合并区
    keep_part = detail_df.iloc[:keep_rows]
    merge_part = detail_df.iloc[keep_rows:]

    # 4. 生成合并行
    merged_lower = merge_part['lower'].min()  # 动态获取 15 行之后的最小下限
    merged_name = f'≥{merged_lower}㎡'  # 自动标签

    merged_row = {
        '面积段': merged_name,
        '供应套数': merge_part['供应套数'].sum(),
        '成交套数': merge_part['成交套数'].sum()
    }
    # 供求比 - 添加除零处理
    if merged_row['成交套数'] > 0:
        merged_row['供求比'] = (merged_row['供应套数'] /
                                merged_row['成交套数']).round(2)
    else:
        merged_row['供求比'] = 0  # 或者使用 float('inf') 表示无穷大

    # 成交占比（对明细合计）
    merged_row['成交占比'] = '{:.1%}'.format(
        merged_row['成交套数'] / detail_df['成交套数'].sum()
    )

    # 5. 重新拼装
    result = pd.concat([
        keep_part.drop(columns='lower'),
        pd.DataFrame([merged_row])
    ], ignore_index=True)

    if not total_row.empty:
        result = pd.concat([result, total_row], ignore_index=True)

    return result

def compact_area_table_1(df, max_rows=16):
    """
       合并房地产面积段数据，将超过最大行数的数据进行合并

       参数:
       df (DataFrame): 包含面积段及对应供应和成交数据的DataFrame
       max_rows (int): 合并后保留的最大行数，默认为15

       返回:
       DataFrame: 合并后的数据表
       """
    df = df[df['面积段'] != '总计']
    # 如果数据行数已经小于等于max_rows，直接返回原数据
    if len(df) <= max_rows:
        return df

    # 复制数据避免修改原始数据
    data = df.copy()

    # 提取面积段的下限值用于排序
    data['lower_bound'] = data.iloc[:, 0].apply(lambda x: float(x.split('-')[0]))

    # 按面积下限排序
    data = data.sort_values('lower_bound').reset_index(drop=True)

    # 保留前(max_rows-1)行
    keep_rows = data.iloc[:max_rows - 1].copy()

    # 将超出的行进行合并
    merge_rows = data.iloc[max_rows - 1:].copy()

    # 获取合并点的面积下限
    merge_point = int(merge_rows['lower_bound'].min())

    # 创建新的合并行
    merged_row = pd.Series(index=data.columns, dtype=object)
    merged_row.iloc[0] = f"≥{merge_point}㎡"

    # 合并数据列(数值列)
    for col in data.columns[1:-1]:  # 不包括第一列(面积段)和最后一列(lower_bound)
        # 确保数据是数值类型
        merged_row[col] = pd.to_numeric(merge_rows[col], errors='coerce').sum()

    # 构建最终结果
    result = pd.concat([
        keep_rows.drop('lower_bound', axis=1),
        pd.DataFrame([merged_row.drop('lower_bound')])
    ], ignore_index=True)

    return result

def compact_area_table_2(df, keep_rows=15, total_label='汇总'):
    """
    当明细行数超过 keep_rows 时，把第 keep_rows+1 行及之后
    的数据合并成一行，面积段名称自动变成 '≥xxx㎡'。
    参数
    ----
    df : DataFrame      原始表
    keep_rows : int     要保留的明细行数，默认 15
    total_label : str   汇总行的标记字段值
    """
    # 0. 拆出汇总行（如果有）
    total_row = df[df['面积段'] == total_label]
    detail_df = df[df['面积段'] != total_label].copy()

    # 1. 取面积段下限，用于排序和后续判断
    def get_lower(area):
        nums = re.findall(r'\d+', str(area))
        return int(nums[0]) if nums else 0

    detail_df['lower'] = detail_df['面积段'].apply(get_lower)
    detail_df = detail_df.sort_values('lower').reset_index(drop=True)

    # 2. 判断是否需要合并
    if len(detail_df) <= keep_rows:
        result = detail_df.drop(columns='lower')
        return (pd.concat([result, total_row], ignore_index=True)
                if not total_row.empty else result)

    # 3. 需要合并：拆分保留区和合并区
    keep_part = detail_df.iloc[:keep_rows]
    merge_part = detail_df.iloc[keep_rows:]

    # 4. 生成合并行
    merged_lower = merge_part['lower'].min()  # 动态获取 15 行之后的最小下限
    merged_name = f'≥{merged_lower}㎡'  # 自动标签

    merged_row = {
        '面积段': merged_name,
        '供应套数': merge_part['供应套数'].sum(),
        '成交套数': merge_part['成交套数'].sum()
    }

    # 5. 重新拼装
    result = pd.concat([
        keep_part.drop(columns='lower'),
        pd.DataFrame([merged_row])
    ], ignore_index=True)

    if not total_row.empty:
        result = pd.concat([result, total_row], ignore_index=True)

    return result

def compact_merge_dataframe_ranges(df, max_rows=10, max_cols=10):
    """
    合并DataFrame中超过指定阈值的行和列

    Parameters:
    df: DataFrame
    max_rows: 保留的最大行数
    max_cols: 保留的最大列数
    """

    def get_merge_label(range_str):
        """从范围字符串生成合并标签"""
        match = re.search(r'(\d+)-(\d+)([^\d]*)', str(range_str))
        if match:
            end_val = match.group(2)
            unit = match.group(3)
            return f"≥{end_val}{unit}"
        return "≥其他"

    result_df = df.copy()

    # 找汇总行和汇总列
    summary_row = None
    summary_col = None

    if '汇总' in result_df.index:
        summary_row = result_df.loc['汇总']
        result_df = result_df.drop('汇总')

    if '汇总' in result_df.columns:
        summary_col = result_df['汇总']
        result_df = result_df.drop('汇总', axis=1)

    # 合并行
    if len(result_df) > max_rows:
        kept_rows = result_df.iloc[:max_rows]
        merged_rows = result_df.iloc[max_rows:]

        merge_label = get_merge_label(kept_rows.index[-1])
        merged_data = merged_rows.sum()
        merged_data.name = merge_label

        result_df = pd.concat([kept_rows, merged_data.to_frame().T])

    # 合并列
    if len(result_df.columns) > max_cols:
        kept_cols = result_df.columns[:max_cols]
        merged_cols = result_df.columns[max_cols:]

        merge_label = get_merge_label(kept_cols[-1])
        merged_data = result_df[merged_cols].sum(axis=1)

        result_df = result_df[kept_cols]
        result_df[merge_label] = merged_data

    # 加回汇总
    if summary_col is not None:
        result_df['汇总'] = result_df.sum(axis=1)

    if summary_row is not None:
        result_df.loc['汇总'] = result_df.sum()

    return result_df

def compact_merge_price_or_area_ranges(df, max_rows=10):
    """
    若DataFrame行数超过max_rows，就只保留前max_rows行，其余合并
    参数:
    df: 区间列为 area_range（或price_range）,
        数量列为 count 的 DataFrame
    max_rows: 最多保留的行数
    """
    result_df = df.copy()

    # 如果有 area_range 列，则用它，否则假设 index 就是区间
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

        # 新合并的一行
        merged_row = pd.DataFrame(
            {label_col: [merge_label], 'count': [merged_count]} if label_col else {'count': [merged_count]},
            index=[kept_df.index[-1]+1 if isinstance(kept_df.index[-1], int) else merge_label]
        )

        # 拼接
        if label_col:
            result_df = pd.concat([kept_df, merged_row], ignore_index=True)
        else:
            result_df = pd.concat([kept_df, merged_row])

    return result_df


