# text_utils.py

import re
from typing import Dict, Any, List, Tuple, Optional


def parse_title_text(text: str) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    """
    从标题文本中解析出年份、板块名称和核心内容。

    Args:
        text: 待解析的文本内容。

    Returns:
        一个元组 (start_year, end_year, block, detail)。
    """
    # 改进的正则表达式，更好地处理 "2021-2023年怀柔区供应与成交趋势" 这种情况
    # 1. ((\d{4})(?:-(\d{4}))?)年: 匹配 "YYYY年" 或 "YYYY-YYYY年"
    # 2. ([^供成]+(?:板块|区|镇|街道))?: 匹配地名，如 "怀柔区板块" 或 "良乡"
    # 3. (.*): 匹配剩余部分作为详情
    match = re.search(r'((\d{4})(?:-(\d{4}))?)年\s*([^供成]+(?:板块|区|镇|街道))?\s*(.*)', text)

    if match:
        full_year_str, start_year, end_year, block, detail = match.groups()
        end_year = end_year or start_year  # 如果没有结束年份，则与开始年份相同
        
        # 清理板块名称，移除"板块"二字
        if block:
            block = block.strip().removesuffix('板块')
            
        return start_year, end_year, block, detail.strip()

    # 如果上述模式不匹配，回退到只查找年份
    years = re.findall(r'\b(\d{4})\b', text)
    start_year = years[0] if years else None
    end_year = years[1] if len(years) > 1 else start_year
    
    return start_year, end_year, None, text.strip()


def extract_details_from_title(title_content: str) -> Dict[str, Any]:
    """
    根据标题内容，提取出数据范围、板块、意图和工具等结构化信息。

    Args:
        title_content: 内容元素的标题字符串。

    Returns:
        一个包含提取信息的字典。
    """
    start_year, end_year, block, detail = parse_title_text(title_content)
    
    info = {}
    
    # 1. 构造 data_range
    if start_year and end_year:
        info["data_range"] = {
            "start_time": f"{start_year}-01-01",
            "end_time": f"{end_year}-12-31"
        }
        
    # # 2. 添加 block
    # if block:
    #     # 与原始YAML格式的 'block' 键匹配
    #     info["block"] = block
    #
    # # 3. 推断 intent
    # intent_list = []
    # if "供应" in title_content:
    #     intent_list.append("supply_sets")
    # if "成交" in title_content:
    #     intent_list.append("trade_sets")
    # if intent_list:
    #     info["intent"] = intent_list
    #
    # # 4. 推断 fun_tool
    # # 将解析出的核心内容作为 fun_tool
    # if detail:
    #     info["fun_tool"] = detail
        
    return info