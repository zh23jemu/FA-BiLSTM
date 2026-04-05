# -*- coding: utf-8 -*-
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Cm, Pt
import os

W, H = 33.867, 19.05

def apply_style(run, size, bold=False, color=None):
    run.font.name = "微软雅黑"
    run.font.size = Pt(size)
    run.font.bold = bold
    if color:
        run.font.color.rgb = RGBColor(*color)

def textbox(slide, l, t, w, h, text, size=12, bold=False, color=(0,0,0), align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE):
    shp = slide.shapes.add_textbox(Cm(l), Cm(t), Cm(w), Cm(h))
    tf = shp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = valign
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    apply_style(r, size, bold, color)
    return shp

def rounded_box(slide, l, t, w, h, fill, border=(100,100,100), title=None, title_size=12, text_color=(0,0,0)):
    shp = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, Cm(l), Cm(t), Cm(w), Cm(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = RGBColor(*fill)
    shp.line.color.rgb = RGBColor(*border)
    shp.line.width = Pt(1.0)
    if title:
        tf = shp.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = title
        apply_style(r, title_size, True, text_color)
    return shp

def arrow(slide, x1, y1, x2, y2, color=(100,100,100), width=1.5, arrowhead=True):
    line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Cm(x1), Cm(y1), Cm(x2), Cm(y2))
    line.line.color.rgb = RGBColor(*color)
    line.line.width = Pt(width)
    if arrowhead:
        line.line.end_arrowhead = True
    return line

def draw_lstm_chain(slide, x_start, y_start, num_cells=4):
    colors = {"fw": (145, 199, 138), "bw": (122, 170, 214)}
    xs = [x_start + i*2.2 for i in range(num_cells)]
    for x in xs:
        # Fw node
        c1 = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.OVAL, Cm(x), Cm(y_start), Cm(0.8), Cm(0.8))
        c1.fill.solid(); c1.fill.fore_color.rgb = RGBColor(*colors["fw"])
        c1.line.color.rgb = RGBColor(255,255,255)
        # Bw node
        c2 = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.OVAL, Cm(x), Cm(y_start+1.5), Cm(0.8), Cm(0.8))
        c2.fill.solid(); c2.fill.fore_color.rgb = RGBColor(*colors["bw"])
        c2.line.color.rgb = RGBColor(255,255,255)
        # Input arrows
        arrow(slide, x+0.4, y_start+3, x+0.4, y_start+2.3)
        arrow(slide, x+0.4, y_start+2.3, x+0.4, y_start+0.8)
    # Horizontal arrows
    for i in range(num_cells-1):
        arrow(slide, xs[i]+0.8, y_start+0.4, xs[i+1], y_start+0.4, color=colors["fw"])
        arrow(slide, xs[i+1], y_start+1.9, xs[i]+0.8, y_start+1.9, color=colors["bw"])
    
    textbox(slide, x_start-0.5, y_start+3, num_cells*2.2, 0.5, "输入序列 Xt = [x_((t-L)), ..., x_t]", size=11, align=PP_ALIGN.CENTER)
    
    # Output h_target
    arrow(slide, xs[-1]+0.8, y_start+0.4, xs[-1]+2.2, y_start+1.1)
    arrow(slide, xs[-1]+0.8, y_start+1.9, xs[-1]+2.2, y_start+1.1)
    rounded_box(slide, xs[-1]+2.2, y_start+0.6, 2.5, 1, (240, 240, 250), title="h_target\n(上下文特征)", title_size=10)

def main():
    prs = Presentation()
    prs.slide_width = Cm(W)
    prs.slide_height = Cm(H)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    
    # Colors
    c_bg = (248, 250, 253)
    bg = slide.background.fill
    bg.solid(); bg.fore_color.rgb = RGBColor(*c_bg)
    
    c_navy = (32, 58, 92)
    c_panel = (236, 242, 249)
    c_line = (116, 143, 176)
    
    # Title
    textbox(slide, 1, 0.5, 31, 1.2, "FA-BiLSTM 测井智能识别架构图", size=24, bold=True, color=c_navy)

    # Panels
    rounded_box(slide, 1, 2.5, 7.5, 15.5, c_panel, c_line)
    textbox(slide, 1, 2.5, 7.5, 1.2, "1. 数据输入 (Data Input)", size=16, bold=True, color=c_navy)
    
    rounded_box(slide, 9, 2.5, 15.5, 15.5, c_panel, c_line)
    textbox(slide, 9, 2.5, 15.5, 1.2, "2. 核心网络 (FA-BiLSTM Engine)", size=16, bold=True, color=c_navy)

    rounded_box(slide, 25, 2.5, 7.8, 15.5, c_panel, c_line)
    textbox(slide, 25, 2.5, 7.8, 1.2, "3. 预测输出 (Prediction Output)", size=16, bold=True, color=c_navy)
    
    # Arrows between panels
    arrow(slide, 8.5, 10, 9, 10, width=3.0)
    arrow(slide, 24.5, 10, 25, 10, width=3.0)

    # ------------------
    # 1. Left Panel
    # ------------------
    textbox(slide, 1.5, 4, 6.5, 1, "多维度连续测井参数", size=14, bold=True, color=(0,0,0))
    rounded_box(slide, 2.2, 5.5, 5, 1, (255,255,255), title="Sigma (俘获截面)", title_size=11)
    rounded_box(slide, 2.2, 7.0, 5, 1, (255,255,255), title="C/O (碳氧比)", title_size=11)
    rounded_box(slide, 2.2, 8.5, 5, 1, (255,255,255), title="Ratio (非弹性比)", title_size=11)
    
    # Slide Window Box
    rounded_box(slide, 1.5, 10.5, 6.5, 6, (230, 240, 255), title="")
    textbox(slide, 1.5, 10.8, 6.5, 1, "基于滑动窗口的时序构造", size=12, bold=True, align=PP_ALIGN.CENTER)
    textbox(slide, 1.8, 12, 5.9, 4, "Window Size (L) = 10\n\n通过沿井深滑动，将单点转化为包含上下文的时序片段：\n(Batch, L, Features)", size=11, align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP)

    # ------------------
    # 2. Middle Panel
    # ------------------
    # Top: BiLSTM
    rounded_box(slide, 9.5, 4, 14.5, 5.5, (255, 255, 255), c_line)
    textbox(slide, 9.5, 4, 14.5, 1, "时序特征提取 (BiLSTM Layer)", size=14, bold=True, color=c_navy)
    draw_lstm_chain(slide, 10.5, 5.2, num_cells=4)
    textbox(slide, 9.5, 8.6, 14.5, 0.8, "说明：双向LSTM分别捕捉“由浅至深”和“由深至浅”的地质趋势", size=11, color=(100,100,100))

    # Bottom: Attention
    rounded_box(slide, 9.5, 10, 14.5, 7.5, (255, 255, 255), c_line)
    textbox(slide, 9.5, 10, 14.5, 1, "特征注意力门控 (Feature Attention Gate)", size=14, bold=True, color=c_navy)
    
    # Attention diagram
    rounded_box(slide, 10, 12, 2.8, 1.2, (217, 238, 243), title="h_target\n(时序上下文表示)", title_size=11)
    rounded_box(slide, 10, 15.5, 2.8, 1.2, (247, 231, 220), title="x_target\n(当前深度物理特征)", title_size=11)
    
    arrow(slide, 12.8, 12.6, 14.5, 13.8)
    arrow(slide, 12.8, 16.1, 14.5, 14.8)
    
    c_att = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.OVAL, Cm(14.5), Cm(13.5), Cm(1.6), Cm(1.6))
    c_att.fill.solid(); c_att.fill.fore_color.rgb = RGBColor(233, 227, 241)
    c_att.line.color.rgb = RGBColor(*c_line)
    textbox(slide, 14.5, 13.5, 1.6, 1.6, "评分\n函数", size=10, bold=True, align=PP_ALIGN.CENTER)
    
    arrow(slide, 16.1, 14.3, 17, 14.3)
    
    rounded_box(slide, 17, 13.5, 2.5, 1.6, (228, 239, 224), title="Sigmoid激活\n分配权重(α)", title_size=11)
    
    arrow(slide, 19.5, 14.3, 20.8, 14.3)
    
    # Route x_target to output
    arrow(slide, 11.4, 16.7, 11.4, 17.1, arrowhead=False)
    arrow(slide, 11.4, 17.1, 21.8, 17.1, arrowhead=False)
    arrow(slide, 21.8, 17.1, 21.8, 15.1)
    
    rounded_box(slide, 20.8, 13.5, 2.5, 1.6, (255, 245, 230), title="特征贡献度加权\nα ⊙ x_target", title_size=11)
    
    textbox(slide, 10, 11.0, 13.5, 1, "逻辑：上下文 h_target 决定当前环境下，各个物理参数 x_target 的贡献大小", size=11, color=(100,100,100), align=PP_ALIGN.LEFT)


    # ------------------
    # 3. Right Panel
    # ------------------
    # Fusion
    rounded_box(slide, 25.5, 3.5, 6.8, 3.5, (255, 255, 255), c_line)
    textbox(slide, 25.5, 3.5, 6.8, 1, "多尺度特征融合", size=14, bold=True, color=c_navy)
    textbox(slide, 25.5, 4.5, 6.8, 2, "拼接上下文与加权特征：\n[ h_target,\n  x_target,\n  weighted_x ]\n送入全连接层", size=10, align=PP_ALIGN.CENTER)
    
    arrow(slide, 28.9, 7, 28.9, 8.5)

    # Output
    rounded_box(slide, 25.5, 8.5, 6.8, 9, (255, 255, 255), c_line)
    textbox(slide, 25.5, 8.5, 6.8, 1.2, "多阶段综合输出", size=14, bold=True, color=c_navy)
    
    # Output items
    rounded_box(slide, 26, 10.5, 5.8, 1.5, (250, 240, 240), title="1. 分类标签 (Gas/Oil/ 0/1)")
    rounded_box(slide, 26, 12.8, 5.8, 1.5, (240, 250, 240), title="2. 储层指数 (平滑概率曲线)")
    rounded_box(slide, 26, 15.1, 5.8, 1.5, (240, 240, 255), title="3. 参数敏感性 (α响应曲线)")

    # Save
    out_file = "FA_BiLSTM_单页架构图_一页版.pptx"
    prs.save(out_file)

if __name__ == "__main__":
    main()
