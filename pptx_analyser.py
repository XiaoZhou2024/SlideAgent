from openai import OpenAI
import json
import re
from typing import List, Dict, Any
import re
from config import config

def _call_vision_model_v2(base64_image: str) -> List[Dict[str, Any]]:
    """
    调用视觉大模型并解析其JSON输出。
    """
    client = OpenAI(
        api_key=config.API_KEY,
        base_url=config.BASE_URL,
    )
    try:
        completion = client.chat.completions.create(
            model="qwen25-vl",
            messages=[{
                "role": "system",
                "content": [{"type": "text", "text": """
                    你是一个PPT分析助手。你的任务是从一张PPT幻灯片图片中，识别出**主标题**、**结论文本**以及**图表/表格的标题**。
                    
                    请严格按照以下JSON格式返回结果。JSON是一个列表，每个元素代表一个识别到的组件。
                    - **shape_type**: 识别到的组件类型。可能的类型为："slide_title"（主标题）, "conclusion"（结论）, "chart_title"（图表标题）, "table_title"（表格标题）, "other_text"（其他文本）。
                    - **content**: 识别到的文本内容。
                    - **bbox**: 识别到的组件在图片中的边界框坐标，格式为 `[x_min, y_min, x_max, y_max]`。
                    
                    如果找不到某个类型的元素，则不要在列表中包含它。请注意，不要返回任何额外的解释或文本，只需返回一个纯粹的JSON数组。
                """}]
            }, {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", "text": "请根据系统提示，分析幻灯片内容，识别主标题、结论和图表/表格标题，并返回JSON。"}
                ],
            }],
        )
        response_content = completion.choices[0].message.content
        if response_content.startswith("```json") and response_content.endswith("```"):
            response_content = response_content[7:-3].strip()
        end_match = re.search(r'\]\s*$', response_content)
        if end_match:
            response_content = response_content[:end_match.end()]
        return json.loads(response_content)
    except Exception as e:
        print(f"调用视觉大模型失败或解析JSON失败: {e}")
        return []
