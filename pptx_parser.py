# pptx_parser.py

import yaml
from pathlib import Path
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.enum.chart import XL_CHART_TYPE # 导入图表类型枚举
from pptx.util import Cm
from math import sqrt
from typing import Dict, List, Any, Tuple

# 从其他模块导入辅助函数
from text_utils import extract_details_from_title

# --- 辅助函数 ---
def emu_to_cm(emu: float) -> float:
    """将EMU（英国测量单位）转换为厘米，并保留两位小数。"""
    return round(emu / 360000.0, 2)

def get_shape_layout(shape) -> Dict[str, float]:
    """从shape对象提取布局信息（位置和尺寸），单位为厘米。"""
    return {
        "x": emu_to_cm(shape.left),
        "y": emu_to_cm(shape.top),
        "width": emu_to_cm(shape.width),
        "height": emu_to_cm(shape.height),
    }

def get_shape_center(shape) -> Tuple[float, float]:
    """计算形状中心的坐标（EMU单位）。"""
    return shape.left + shape.width / 2, shape.top + shape.height / 2

# --- 主解析器类 ---
class PptxParser:
    """
    一个用于解析PPTX文件幻灯片并提取其结构化信息的类。
    """
    def __init__(self, pptx_path: Path):
        """
        初始化解析器。

        Args:
            pptx_path: 要解析的PPTX文件的路径。
        """
        if not pptx_path.exists():
            raise FileNotFoundError(f"PPT文件未找到: {pptx_path}")
        self.presentation = Presentation(pptx_path)
        self.file_name = pptx_path.stem

    def _get_shape_type(self, shape) -> str:
        """
        获取形状的规范化类型名称。
        """
        if shape.shape_type == MSO_SHAPE_TYPE.TABLE:
            return "table"
        
        if shape.shape_type == MSO_SHAPE_TYPE.CHART:
            if hasattr(shape, "chart"):
                chart_type = shape.chart.chart_type
                # 柱状图和条形图
                if chart_type in (
                    XL_CHART_TYPE.COLUMN_CLUSTERED, XL_CHART_TYPE.COLUMN_STACKED, XL_CHART_TYPE.COLUMN_STACKED_100,
                    XL_CHART_TYPE.BAR_CLUSTERED, XL_CHART_TYPE.BAR_STACKED, XL_CHART_TYPE.BAR_STACKED_100
                ):
                    return "chart-bar"
                # 折线图
                elif chart_type in (
                    XL_CHART_TYPE.LINE, XL_CHART_TYPE.LINE_MARKERS, XL_CHART_TYPE.LINE_STACKED, 
                    XL_CHART_TYPE.LINE_STACKED_100, XL_CHART_TYPE.LINE_MARKERS_STACKED, XL_CHART_TYPE.LINE_MARKERS_STACKED_100
                ):
                    return "chart-line"
                else:
                    # 其他图表类型的回退选项
                    return "chart-other"
            return "chart"

        if shape.has_text_frame and shape.text.strip():
            return "text"
            
        return "other"

    def _classify_shapes(self, slide) -> Dict[str, List[Dict[str, Any]]]:
        """
        遍历幻灯片上的所有形状，并将其分类。
        """
        classified_shapes = {
            "text": [], "table": [], "chart-bar": [], "chart-line": [], 
            "chart-other": [], "other": []
        }
        
        for shape in slide.shapes:
            shape_type = self._get_shape_type(shape)
            if shape_type == "other":
                continue

            shape_info = {
                "layout": get_shape_layout(shape),
                "center": get_shape_center(shape),
                "obj": shape
            }
            if shape_type == "text":
                shape_info["content"] = shape.text.strip()
            
            # 确保键存在
            if shape_type not in classified_shapes:
                classified_shapes[shape_type] = []
            classified_shapes[shape_type].append(shape_info)
        
        # 按垂直位置排序文本框
        classified_shapes["text"].sort(key=lambda s: s["layout"]["y"])
        return classified_shapes

    def parse_slide(self, slide_idx: int = 0) -> Dict[str, Any]:
        """
        解析指定索引的幻灯片，并生成与目标YAML格式匹配的字典。

        Args:
            slide_idx: 要解析的幻灯片的索引（从0开始）。

        Returns:
            一个代表幻灯片结构的字典。
        """
        if slide_idx >= len(self.presentation.slides):
            raise IndexError(f"幻灯片索引 {slide_idx} 超出范围。")
        
        slide = self.presentation.slides[slide_idx]
        
        # 1. 对幻灯片上的所有形状进行分类
        shapes = self._classify_shapes(slide)
        text_shapes = shapes["text"]
        content_elements_shapes = (
            shapes["table"] + shapes["chart-bar"] + 
            shapes["chart-line"] + shapes["chart-other"]
        )

        # 2. 识别幻灯片标题和分析文本
        slide_title, analysis_text, element_titles = None, None, []
        if len(text_shapes) >= 1:
            slide_title = {"content": text_shapes[0]["content"], "layout": text_shapes[0]["layout"]}
        if len(text_shapes) >= 2:
            analysis_text = {"content": text_shapes[1]["content"], "layout": text_shapes[1]["layout"]}
            element_titles = text_shapes[2:]
        elif len(text_shapes) == 1:
            element_titles = text_shapes[1:]

        # 3. 构建 content_elements
        content_elements = []
        for title_info in element_titles:
            if not content_elements_shapes:
                break
            
            closest_element = min(
                content_elements_shapes,
                key=lambda el: sqrt(
                    (el["center"][0] - title_info["center"][0])**2 +
                    (el["center"][1] - title_info["center"][1])**2
                )
            )
            
            details = extract_details_from_title(title_info["content"])
            
            element = {
                "title": {"content": title_info["content"], "layout": title_info["layout"]},
                "shape_type": self._get_shape_type(closest_element["obj"]),
                "layout": closest_element["layout"],
                **details
            }
            content_elements.append(element)
            content_elements_shapes.remove(closest_element)

        # 4. 组装最终的字典结构
        final_structure = {
            "template_slide": {
                "slide_size": {"width": emu_to_cm(self.presentation.slide_width), "height": emu_to_cm(self.presentation.slide_height)},
                "title": slide_title,
                "analysis": analysis_text,
                "content_elements": content_elements
            }
        }
        return final_structure

    @staticmethod
    def save_dict_as_yaml(data: Dict, output_path: Path):
        """将字典保存为YAML文件。"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, indent=2, sort_keys=False)
        print(f"成功将提取的结构保存到: {output_path}")
