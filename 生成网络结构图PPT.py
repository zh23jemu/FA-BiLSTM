# -*- coding: utf-8 -*-
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Cm, Pt


COLORS = {
    "navy": (28, 52, 84),
    "ink": (46, 56, 66),
    "muted": (99, 112, 126),
    "line": (120, 130, 142),
    "panel": (247, 249, 252),
    "mint": (223, 241, 236),
    "sand": (251, 237, 214),
    "blue": (222, 234, 248),
    "green": (224, 238, 214),
    "rose": (244, 225, 224),
    "lavender": (234, 226, 242),
    "gold": (252, 240, 206),
    "silver": (232, 234, 238),
}


def set_text_style(run, font_size, bold=False, color=None, name="Microsoft YaHei"):
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.name = name
    if color:
        run.font.color.rgb = RGBColor(*color)


def add_textbox(slide, left, top, width, height, text, font_size=12, bold=False,
                color=None, align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP):
    shape = slide.shapes.add_textbox(Cm(left), Cm(top), Cm(width), Cm(height))
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.vertical_anchor = valign
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    set_text_style(run, font_size, bold, color or COLORS["ink"])
    return shape


def add_box(slide, left, top, width, height, fill, title, subtitle=None,
            title_size=18, sub_size=10, line=None):
    shape = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE,
        Cm(left), Cm(top), Cm(width), Cm(height)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(*fill)
    shape.line.color.rgb = RGBColor(*(line or COLORS["line"]))
    shape.line.width = Pt(1.2)

    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE

    p1 = tf.paragraphs[0]
    p1.alignment = PP_ALIGN.CENTER
    r1 = p1.add_run()
    r1.text = title
    set_text_style(r1, title_size, True, COLORS["navy"])

    if subtitle:
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        r2 = p2.add_run()
        r2.text = subtitle
        set_text_style(r2, sub_size, False, COLORS["muted"])
    return shape


def add_badge(slide, cx, cy, text, fill):
    shape = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.OVAL,
        Cm(cx - 0.45), Cm(cy - 0.45), Cm(0.9), Cm(0.9)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(*fill)
    shape.line.color.rgb = RGBColor(255, 255, 255)
    shape.line.width = Pt(1.0)
    tf = shape.text_frame
    tf.clear()
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = text
    set_text_style(r, 11, True, (255, 255, 255))
    return shape


def add_arrow(slide, x1, y1, x2, y2, color=None, width=2.2):
    line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Cm(x1), Cm(y1), Cm(x2), Cm(y2))
    line.line.color.rgb = RGBColor(*(color or COLORS["line"]))
    line.line.width = Pt(width)
    line.line.end_arrowhead = True
    return line


def add_panel(slide, left, top, width, height, title, body, fill):
    panel = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE,
        Cm(left), Cm(top), Cm(width), Cm(height)
    )
    panel.fill.solid()
    panel.fill.fore_color.rgb = RGBColor(*fill)
    panel.line.color.rgb = RGBColor(*COLORS["line"])
    panel.line.width = Pt(1.0)

    add_textbox(slide, left + 0.4, top + 0.35, width - 0.8, 0.6, title,
                font_size=16, bold=True, color=COLORS["navy"])
    add_textbox(slide, left + 0.4, top + 1.1, width - 0.8, height - 1.4, body,
                font_size=11, color=COLORS["ink"])
    return panel


prs = Presentation()
prs.slide_width = Cm(33.867)
prs.slide_height = Cm(19.05)
slide = prs.slides.add_slide(prs.slide_layouts[6])

# Background
bg = slide.background.fill
bg.solid()
bg.fore_color.rgb = RGBColor(255, 255, 255)

# Header
slide.shapes.add_shape(
    MSO_AUTO_SHAPE_TYPE.RECTANGLE, Cm(0), Cm(0), Cm(33.867), Cm(1.35)
).fill.solid()
header_bar = slide.shapes[0]
header_bar.fill.fore_color.rgb = RGBColor(*COLORS["navy"])
header_bar.line.fill.background()

add_textbox(slide, 0.9, 0.22, 11.0, 0.8, "FA-BiLSTM 网络结构图",
            font_size=25, bold=True, color=(255, 255, 255), align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.MIDDLE)
add_textbox(slide, 21.0, 0.28, 11.8, 0.6, "输入序列  ->  时序编码  ->  特征耦合  ->  分类输出",
            font_size=11, color=(229, 236, 244), align=PP_ALIGN.RIGHT, valign=MSO_ANCHOR.MIDDLE)

# Main pipeline
boxes = [
    (1.2, 3.0, 4.6, 2.2, COLORS["mint"], "输入序列", "窗口长度 = 10"),
    (7.0, 3.0, 4.6, 2.2, COLORS["sand"], "BiLSTM 编码", "1层双向LSTM | hidden = 16"),
    (12.8, 1.95, 4.2, 1.75, COLORS["blue"], "上下文向量", "h_target"),
    (12.8, 4.35, 4.2, 1.75, COLORS["blue"], "当前特征", "x_target"),
    (18.3, 3.0, 4.9, 2.2, COLORS["green"], "特征注意力", "attention_dim = 32"),
    (24.4, 1.95, 4.2, 1.75, COLORS["rose"], "加权特征", "weighted_x = α ⊙ x"),
    (24.4, 4.35, 4.2, 1.75, COLORS["lavender"], "特征融合", "[h, x, weighted_x]"),
    (29.6, 3.0, 3.1, 2.2, COLORS["gold"], "分类头", "MLP + Dropout"),
]

for idx, (l, t, w, h, fill, title, sub) in enumerate(boxes, 1):
    add_box(slide, l, t, w, h, fill, title, sub)
    if idx <= 6:
        badge_x = l + 0.45
        badge_y = t - 0.15
        add_badge(slide, badge_x, badge_y, str(idx), COLORS["navy"])

# labels under pipeline
add_textbox(slide, 1.3, 5.45, 4.8, 0.9,
            "有效储层: MLR, AMPST, PHIE\n气液识别: Sigma, RICX, RIN13, RATO13",
            font_size=10, color=COLORS["muted"], align=PP_ALIGN.CENTER)
add_textbox(slide, 7.2, 5.45, 4.3, 0.7,
            "提取井深方向的上下文时序特征", font_size=10, color=COLORS["muted"], align=PP_ALIGN.CENTER)
add_textbox(slide, 18.4, 5.45, 4.9, 0.9,
            "由 h_target 与 x_target 共同计算\n各输入参数的动态权重 α", font_size=10, color=COLORS["muted"], align=PP_ALIGN.CENTER)
add_textbox(slide, 24.2, 6.1, 8.6, 0.7,
            "Linear(fusion_size -> 16) + ReLU + Dropout(0.2) + Linear(16 -> 2)",
            font_size=10, color=COLORS["muted"], align=PP_ALIGN.CENTER)

# Connectors
add_arrow(slide, 5.8, 4.1, 7.0, 4.1)
add_arrow(slide, 11.6, 4.1, 12.8, 2.85)
add_arrow(slide, 11.6, 4.1, 12.8, 5.2)
add_arrow(slide, 17.0, 2.85, 18.3, 4.0)
add_arrow(slide, 17.0, 5.2, 18.3, 4.2)
add_arrow(slide, 23.2, 4.0, 24.4, 2.85)
add_arrow(slide, 23.2, 4.2, 24.4, 5.2)
add_arrow(slide, 28.6, 5.2, 29.6, 4.1)

# Output + cards area
add_panel(
    slide, 1.2, 9.2, 7.0, 4.6, "输出结果",
    "1. 分类标签\n2. 连续指示曲线（Logits / 归一化曲线）\n3. 动态特征权重曲线 FA_alpha_*",
    COLORS["silver"]
)

add_panel(
    slide, 9.0, 9.2, 8.3, 4.6, "训练设置",
    "学习率 0.003\n最大轮数 180\nEarly Stopping = 25\nAdam + weight decay = 1e-4\n高斯噪声增强 + 少量特征 dropout\n梯度裁剪 max_norm = 3.0",
    (225, 236, 243)
)

add_panel(
    slide, 18.1, 9.2, 6.8, 4.6, "模块含义",
    "BiLSTM 负责提取井深方向的时序依赖。\n特征注意力层负责多参数动态耦合。\n融合层同时保留时序信息与原始物理特征。",
    (247, 232, 229)
)

add_panel(
    slide, 25.6, 9.2, 7.1, 4.6, "消融实验",
    "LDA：线性基线\nBiLSTM：仅时序建模\nFA-BiLSTM：时序建模 + 特征注意力\n用于验证鲁棒性与多参数耦合增益",
    (232, 240, 219)
)

# subtle section captions
add_textbox(slide, 1.2, 8.3, 6.0, 0.5, "模型输出与实验说明", font_size=12, bold=True, color=COLORS["navy"])
add_textbox(slide, 1.2, 1.75, 8.0, 0.5, "主流程", font_size=12, bold=True, color=COLORS["navy"])

output_path = "FA_BiLSTM_网络结构图_优化版.pptx"
prs.save(output_path)
print(output_path)
