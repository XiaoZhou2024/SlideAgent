import re
from typing import Dict, Any, List, Tuple, Optional


def parse_title_text(text: str) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    match = re.search(r'((\d{4})(?:-(\d{4}))?)?\s*(.*)', text)

    if match:
        full_year_str, start_year, end_year, block, detail = match.groups()
        end_year = end_year or start_year

        return start_year, end_year, block, detail.strip()

    years = re.findall(r'\b(\d{4})\b', text)
    start_year = years[0] if years else None
    end_year = years[1] if len(years) > 1 else start_year
    
    return start_year, end_year, None, text.strip()


def extract_details_from_title(title_content: str) -> Dict[str, Any]:
    start_year, end_year, block, detail = parse_title_text(title_content)
    
    info = {}

    if start_year and end_year:
        info["data_range"] = {
            "start_time": f"{start_year}-01-01",
            "end_time": f"{end_year}-12-31"
        }

    return info