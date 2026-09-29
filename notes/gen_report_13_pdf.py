﻿# -*- coding: utf-8 -*-
# 生成 13_BEV模型调试实战复盘 PDF（图片 + 黄色荧光标注 + 防重叠排版）
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak, KeepTogether)
from PIL import Image as PILImage
import os

BASE = r"c:\Users\PC\Documents\trae_projects\bev"
VIS = os.path.join(BASE, "vis_results")
OUT = os.path.join(BASE, "notes", "13_BEV模型调试实战复盘_20260929.pdf")

# ---------- 字体 ----------
def register_fonts():
    candidates = [
        ("RptFont", r"C:\Windows\Fonts\msyh.ttc", 0),
        ("RptFont-Bold", r"C:\Windows\Fonts\msyhbd.ttc", 0),
    ]
    ok = {}
    for name, path, idx in candidates:
        try:
            pdfmetrics.registerFont(TTFont(name, path, subfontIndex=idx))
            ok[name] = True
        except Exception:
            try:
                alt = r"C:\Windows\Fonts\simhei.ttf" if "Bold" in name else r"C:\Windows\Fonts\simsun.ttc"
                pdfmetrics.registerFont(TTFont(name, alt, subfontIndex=0))
                ok[name] = True
            except Exception:
                ok[name] = False
    return ok

FONTS = register_fonts()
F = "RptFont" if FONTS.get("RptFont") else "Helvetica"
FB = "RptFont-Bold" if FONTS.get("RptFont-Bold") else F

# ---------- 颜色 ----------
HL_BG = colors.HexColor("#FFF200")   # 黄色荧光笔
INK = colors.HexColor("#222222")
GREY = colors.HexColor("#666666")
HEAD_BG = colors.HexColor("#E8F0FE")
LINE = colors.HexColor("#BBBBBB")

PAGE_W, PAGE_H = A4
ML = MR = 1.9 * cm
MT = 1.8 * cm
MB = 1.8 * cm
AVAIL_W = PAGE_W - ML - MR  # ~495pt

# ---------- 样式（leading >= 1.5x 字号，防重叠核心） ----------
def st(name, size, leading=None, font=None, color=INK, align=TA_LEFT,
       before=0, after=0, indent=0):
    return ParagraphStyle(name, fontName=font or F, fontSize=size,
                          leading=leading or size * 1.55, textColor=color,
                          alignment=align, spaceBefore=before, spaceAfter=after,
                          leftIndent=indent, allowWidows=0, allowOrphans=0)

S_TITLE = st("t", 20, 27, font=FB, align=TA_CENTER, after=6)
S_META = st("meta", 9.5, 15, color=GREY, align=TA_CENTER, after=14)
S_H1 = st("h1", 14.5, 21, font=FB, before=14, after=8)
S_H2 = st("h2", 12, 18, font=FB, before=10, after=6)
S_P = st("p", 10, 16, after=5)
S_QUOTE = st("q", 10, 16.5, color=colors.HexColor("#1a5276"), after=5,
             indent=10)
S_CAP = st("cap", 9, 13.5, color=GREY, align=TA_CENTER, after=4)
S_CELL = st("cell", 8.8, 13.2)
S_CELL_H = st("cellh", 9, 13.5, font=FB)
S_LI = st("li", 10, 16, after=3, indent=14)

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def hl(s):
    """转义后把 ==xx== 变成黄色荧光标注"""
    s = esc(s)
    while "==" in s:
        s = s.replace("==", '<font backColor="#FFF200">', 1) \
             .replace("==", "</font>", 1)
    return s

def P(text, style=S_P):
    return Paragraph(hl(text), style)

def H1(t):
    return Paragraph(esc(t), S_H1)

def H2(t):
    return Paragraph(esc(t), S_H2)

def bullets(items):
    return [Paragraph("• " + hl(t), S_LI) for t in items]

def make_table(headers, rows, col_ratios):
    widths = [AVAIL_W * r for r in col_ratios]
    data = [[Paragraph(esc(h), S_CELL_H) for h in headers]]
    for row in rows:
        data.append([Paragraph(hl(c), S_CELL) for c in row])
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
        ("GRID", (0, 0), (-1, -1), 0.5, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t

def img_block(path, caption, width):
    """图片 + 图注，绑定不跨页断开"""
    full = os.path.join(VIS, path)
    with PILImage.open(full) as im:
        w, h = im.size
    w_pt = min(width, AVAIL_W)
    h_pt = w_pt * h / w
    img = Image(full, width=w_pt, height=h_pt)
    img.hAlign = "CENTER"
    cap = Paragraph(esc(caption), S_CAP)
    return KeepTogether([Spacer(1, 4), img, Spacer(1, 3), cap, Spacer(1, 6)])

def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont(F, 8)
    canvas.setFillColor(GREY)
    canvas.drawString(ML, PAGE_H - 1.1 * cm, "BEVFormer 调试实战复盘 · 2026-09-29")
    canvas.drawRightString(PAGE_W - MR, 1.1 * cm, f"第 {doc.page} 页")
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.4)
    canvas.line(ML, 1.35 * cm, PAGE_W - MR, 1.35 * cm)
    canvas.restoreState()

story = []

# ================= 封面区 =================
story.append(Spacer(1, 10))
story.append(Paragraph("BEVFormer 模型调试实战复盘", S_TITLE))
story.append(Paragraph("从「heatmap 全紫、检测无框」到「9 张图全部命中 GT」 · 2026-09-29", S_META))

story.append(H1("一、今天做了什么"))
story.append(P("模型训练“看起来在跑”但可视化全错：heatmap 全紫、检测图无框。今天用一天时间挖出了 ==6 层嵌套的根因==，逐层修复后重训 30 epoch，cls loss 从 11.67 降到 0.27，9 张可视化图全部达标。"))
story.append(make_table(
    ["指标", "修复前", "修复后"],
    [["cls loss", "卡死在 **0.765**（不可学任务的理论最优值）".replace("**", ""), "**0.27**，持续下降"],
     ["heatmap", "全紫一片，无任何亮斑", "亮斑精确落在 GT 物体位置"],
     ["检测图", "无框或一堆乱框", "一框对一物，分数 0.44 ~ 0.71"],
     ["框数量", "14 / 19 / 12 / 15（大量重复）", "4 / 4 / 5 / 4（恰好等于物体数）"]],
    [0.18, 0.44, 0.38]))
story.append(Spacer(1, 4))
story.append(P("==核心认知：模型不收敛 ≠ 调参问题，先怀疑“任务本身不可学”。== 当 loss 卡在一个“奇怪的稳定值”，它往往不是噪声，而是模型对当前任务的理论最优解 —— 这是破案的第一条线索。", S_QUOTE))

# ================= 二、数值破案 =================
story.append(H1("二、数值破案：两条铁证"))
story.append(H2("2.1 铁证一：cls = 0.765 是“不可学任务”的理论最优值"))
story.append(P("训练日志里 cls loss 长期卡在 0.765 不动。用 Focal Loss 的数学性质反推："))
story.extend(bullets([
    "数据集正负样本比约 1:40，Focal Loss 下“全图输出同一个低概率背景值”就能把 loss 压到一个局部最优；",
    "反推该值对应每个网格的 sigmoid 输出 ≈ 0.04（全图统一输出背景概率 4%）；",
    "==模型“看穿”了图像里没有可用于定位物体的信息==，干脆输出常数 —— 这正是“图像与标签无几何对应”时的理论最优解。",
]))
story.append(H2("2.2 铁证二：reg ≡ 1.0 是“均匀分布”的指纹"))
story.append(P("回归 loss 恒等于 1.0，一点不降。L1 Loss 对均匀分布预测的期望值正好是 1.0 —— 说明模型输出的中心偏移量完全是“瞎猜”。==分类和回归同时退化为常数输出，说明病灶不在检测头，而在上游特征根本没传过来。=="))
story.append(P("学习点：训练日志不是“等它降就完事”的进度条，而是模型的体检报告。看到反常的稳定值，先算一算它的解析含义。"))

# ================= 三、六层根因 =================
story.append(PageBreak())
story.append(H1("三、六层根因逐层拆解"))
story.append(P("按挖掘顺序（也是依赖顺序）整理。每层独立看都是“小 bug”，叠在一起就是“完全学不动”。"))

def bug_block(title, rows):
    story.append(H2(title))
    story.extend([Paragraph(f"<b>{k}</b>：{hl(v)}", S_P) for k, v in rows])

bug_block("Bug 1 · 数据不可学（根因中的根因）", [
    ("现象", "模型怎么训都收敛到常数输出。"),
    ("根因", "原 dataset.py 中图像和 GT 热图是两次独立随机生成 —— 图像上的色块和 GT 框的位置毫无几何关系，等于让模型从噪声里预测另一份噪声。"),
    ("修复", "新增 _generate_scene_objects()，每帧先生成 3~8 个物体（类别/位置/尺寸），==图像和 GT 共用同一份物体数据源==；图像用与 SCA 完全一致的投影代码 project_bev_to_image 把物体画成色块（近大远小、按类别着色）。"),
    ("教训", "==投影一致性必须贯穿“数据生成 → SCA 采样 → 可视化”全链路==，任何一处用了不同的几何假设，模型学的就是错误对应。"),
])
bug_block("Bug 2 · meshgrid 顺序转置（坐标约定不一致）", [
    ("现象", "即使数据对了，定位信号仍然错位。"),
    ("根因", "SCA 与位置编码用 meshgrid(z, y, x)（y-major 展平），检测头把 BEV 特征 view(B,H,W,C)（x-major 展平）。同一个 query 在两边对应物理空间中不同的点，位置编码等于全部加错。"),
    ("修复", "全链路统一为 x-major：torch.meshgrid(xs, ys, zs, indexing=\"ij\")，并用 6 个抽样索引做数值验证。"),
    ("教训", "==展平顺序是 BEV 系统最隐蔽的 bug 源，必须全工程统一并在关键模块间做数值对齐测试。=="),
])
bug_block("Bug 3 · grid_sample 归一化错误（97% 采样点越界）", [
    ("现象", "SCA 采样到的图像特征几乎全 0。"),
    ("根因", "把原图像素坐标除以了特征图宽度(8)而不是原图宽高(256)，“归一化坐标”范围成了 [-1, 63]，而 F.grid_sample 只认 [-1,1] —— 97% 的采样点落在界外，采回来全是 0。"),
    ("修复", "grid_u = 2 * pix_x / img_w - 1，并把 img_h / img_w 显式透传到 SCA 层。实测有效采样率从 3.1% 恢复到 100%。"),
    ("教训", "==grid_sample 的坐标是归一化到 [-1,1] 的原图坐标，除数永远是原图尺寸，不是特征图尺寸。=="),
])
story.append(PageBreak())
bug_block("Bug 4 · TSA 无历史帧时的全局自平均", [
    ("现象", "第一帧（没有任何历史 BEV）信号被抹平。"),
    ("根因", "简化版 TSA 用全连接 self-attention 代替原版 deformable attention。无历史帧时 K/V 退化为当前帧自身，2500 个 query 互相平均，把 SCA 刚建立的局部定位信号稀释成均匀背景。"),
    ("修复", "无历史帧时恒等返回；有历史帧时 K/V 只取历史帧（不拼当前帧）。"),
    ("教训", "==用标准 attention 替代 deformable attention 时，要清楚替代品改变了“每个 query 看哪里”的归纳偏置== —— 原版 query 只看自己空间对应的位置，全连接版会全局平均。"),
])
bug_block("Bug 5 · Post-LN 破坏加性信号（最深层、最反直觉）", [
    ("现象", "消融实验：裸 SCA 输出前景 0.178 / 背景 0.018（信号清晰），==只加一层 LayerNorm 就崩到 0.052 / 0.040==。"),
    ("根因", "Post-LN（残差相加后再归一化）对每个 token 重新缩放。本模型是“强随机 query + 小幅加性物体信号 δ”的设计，Post-LN 把 δ 连同残差一起归一化掉，破坏了“物体位置数值偏大”的线性可分性。"),
    ("修复", "改为 Pre-LN：归一化只作用于注意力/FFN 的输入分支，残差主路径保持原始尺度。"),
    ("教训", "==Pre-LN vs Post-LN 不是风格问题，是“残差主路径保不保真”的问题==。现代 DETR 系（含 BEVFormer 原版）全部用 Pre-LN 变体不是偶然。"),
])
bug_block("Bug 6 · 两个辅助病灶", [
    ("CNN 看不见色块", "原背景 ±0.2 强噪声下，色块下采样后能量比仅 1.08。修复：背景改平缓灰调（±0.03）、色块高对比着色加描边。==人眼一眼看不出的目标，CNN 更看不出==。"),
    ("采样冷启动死锁", "所有 offset 初始为 0，9 个采样点重叠在一个像素上，小目标永远采不到、offset 永远学不动。修复：bias 初始化为 3×3 散开网格（±22px）。==初始化要保证“第一步就有信息流”==。"),
])

# ================= 四、验证方法论 =================
story.append(H1("四、验证方法论：不猜，用实验说话"))
story.append(make_table(
    ["步骤", "手段", "通过标准", "实测结果"],
    [["1", "数据冒烟测试", "投影颜色命中 GT 位置", "16/17 命中"],
     ["2", "展平顺序数值验证", "SCA/head 索引逐点对齐", "6/6 索引全对"],
     ["3", "采样有效性探针", "归一化坐标在 [-1,1]", "3.1% → 100%"],
     ["4", "过拟合实验（2 序列 × 150 步）", "前景/背景概率显著分离", "0.149 vs 0.0118（12.6 倍）"],
     ["5", "组件消融（裸 SCA → +LN → +FFN）", "定位“哪一层加进去崩的”", "+一层 LN 即崩"],
     ["6", "全量重训 30 epoch", "cls 显著跌破 0.765", "0.27"]],
    [0.08, 0.34, 0.31, 0.27]))
story.append(Spacer(1, 4))
story.append(P("==过拟合实验 = 最小可复现实验==：全量数据训不动时，先在 2 个样本上过拟合。2 个样本都过拟合不了的模型，100 个样本更不可能学好；反之若 2 个样本能过拟合而全量不行，差异就在数据多样性/正则化。==先把链路在最小规模上打通，再扩大规模==。", S_QUOTE))

# ================= 五、9 张图 =================
story.append(PageBreak())
story.append(H1("五、最终结果：9 张图逐图讲解"))

story.append(H2("5.1 热图（4 张）：模型“看哪里”"))
story.append(P("热图是 cls 分支经 sigmoid 后的概率图，颜色越亮 = 模型认为该 BEV 网格存在该类物体的置信度越高。==“该亮的亮、该暗的暗”是分类分支学对的第一标准==。"))
story.append(img_block("frame0_heatmap.png", "Frame 0：car 通道亮斑 (−15,19)、(3,19) 对准两辆 car GT；ped 亮斑 (−11,−17)；cyclist 通道全暗（本帧无骑行者）", 465))
story.append(img_block("frame1_heatmap.png", "Frame 1：car (7,13)、ped (5,9)、cyclist (−14,−7)，三个类别三个位置互不串扰", 465))
story.append(img_block("frame2_heatmap.png", "Frame 2：car 通道全暗（无 car GT），cyclist (6,3)、ped (−1,−11) 与 GT 一一对应。背景 X 形纹路是位置编码残差，幅度 ≤0.1 不影响判读", 465))
story.append(img_block("frame3_heatmap.png", "Frame 3：ped (9,17)、(7,0) 与 cyclist (7,5)，全部对准 GT", 465))

story.append(PageBreak())
story.append(H2("5.2 检测图（4 张）：模型“画什么”"))
story.append(P("黑色虚线 = GT，红/绿/蓝实线 = 预测（car / ped / cyclist），数字 = 置信度。"))
story.append(PageBreak())
story.append(img_block("frame0_detection.png", "Frame 0：car 0.54、car 0.60、ped 0.44 三框全部套住 GT 虚线框，无重复框、无跨类别误报", 320))
story.append(img_block("frame1_detection.png", "Frame 1：car 0.5~0.7、ped 0.69、cyclist 0.52，一框一物", 320))
story.append(PageBreak())
story.append(img_block("frame2_detection.png", "Frame 2：ped 0.59 / 0.71、cyclist 0.45 / 0.46 对准 GT", 320))
story.append(img_block("frame3_detection.png", "Frame 3：ped 0.47 × 2、cyclist 0.5 × 2，密集小目标全部分开", 320))

story.append(PageBreak())
story.append(H2("5.3 投影图（1 张）：SCA 的“眼睛”"))
story.append(img_block("frame0_projection.png", "2500 个 BEV query 的 3D reference point 投影到各相机，呈放射状扇形铺开", 465))
story.append(P("==每一条红色射线就是“某个 BEV 网格在某个相机里可能看到的位置”==。SCA 做的事就是沿这些点采样图像特征并加权聚合回 BEV —— 投影图正常，是 SCA 能学到东西的几何前提。"))

story.append(H2("5.4 最后一步：NMS 后处理"))
story.append(P("解码时每个物体周围的多个网格都会放电（各 0.1~0.6 分），直接画出来是一堆叠框。加入==类别无关 NMS（IoU > 0.25 抑制）+ 阈值 0.3==后，框数从 14/19/12/15 收敛到 4/4/5/4。==NMS 是几乎所有检测器（含 BEVFormer 原版）的标准后处理，属于解码逻辑而非“改分数”==。"))

# ================= 六、知识点 =================
story.append(PageBreak())
story.append(H1("六、知识点清单（今日新增）"))
story.append(make_table(
    ["#", "知识点", "一句话理解"],
    [["1", "Focal Loss 数值指纹", "loss 卡在反常稳定值 = 模型输出的解析解，可反推任务可学性"],
     ["2", "数据-标签几何一致性", "图像与 GT 必须源自同一份物体数据 + 同一套投影代码"],
     ["3", "meshgrid 展平顺序", "x-major vs y-major 差一个转置，全链路必须统一"],
     ["4", "grid_sample 坐标系", "[-1,1] 归一化的除数是原图尺寸，不是特征图尺寸"],
     ["5", "Self-Attention 归纳偏置", "全连接 attention 会全局平均，deformable 保持局部性"],
     ["6", "Pre-LN vs Post-LN", "小幅加性信号场景下 Post-LN 抹平信号，必须 Pre-LN"],
     ["7", "过拟合实验", "2 样本 × 150 步是验证链路的最小可复现实验"],
     ["8", "组件消融", "逐层加回组件，定位“哪一层引入病灶”"],
     ["9", "采样冷启动", "offset 全零初始化 = 死锁，需散开初始化保证首步信息流"],
     ["10", "NMS", "类别无关 NMS 去跨类重复框，是检测标准后处理"]],
    [0.06, 0.30, 0.64]))

# ================= 七、学习建议 =================
story.append(H1("七、学习建议"))
story.append(H2("7.1 这次调试暴露的短板与对策"))
story.append(make_table(
    ["短板", "证据", "对策"],
    [["数值敏感度不足", "cls=0.765 / reg=1.0 两个铁证摆了几天没被解读", "每次训练先记 loss 初值与稳定值，学会算“理论最优解”"],
     ["独立调试信心弱", "中途两次想回退到“更早的坏版本”", "建立“先实验后动手”清单：先跑探针，再改代码"],
     ["坐标系直觉弱", "meshgrid / grid_sample 两处坐标 bug", "手推“物理坐标 → 像素坐标 → 归一化坐标”全链路并默写"]],
    [0.22, 0.40, 0.38]))
story.append(H2("7.2 下一步（1~2 周内）"))
story.extend(bullets([
    "默写链条：不看代码，手写“BEV query → SCA 采样 → head 解码”的 shape 变换链，写不出来说明还没真懂；",
    "复现消融：把 Post-LN 消融实验自己从零搭一遍（裸 SCA / +LN / +FFN 三版），数值复现“加一层 LN 就崩”；",
    "读原版对照：在 mmdet3d 0.x 分支 BEVFormer 源码中找出今天 6 个 bug 各自对应的“原版正确写法”，做对照笔记；",
    "加分项：给 NMS 写一个旋转框（OBB）版本，为后续真数据做准备。",
]))
story.append(H2("7.3 面试话术（可直接背）"))
story.append(P("“我做过一个 BEVFormer 简化实现的完整调试：模型不收敛，我从 loss 的稳定值反推出任务是‘不可学’的，然后==用过拟合实验 + 组件消融逐层定位出 6 个嵌套 bug==——数据几何不一致、meshgrid 展平顺序、grid_sample 归一化、TSA 全局平均、Post-LN 抹平加性信号、采样冷启动。==最终 cls 从 0.765 降到 0.27==，可视化亮斑与 GT 精确对齐。这段经历让我养成了==‘先算理论值、再做最小实验、最后改代码’==的调试习惯。”"))
story.append(Spacer(1, 2))
story.append(Paragraph("这段话覆盖面试官最想听的三个维度：<b>定位问题的方法论、对 transformer 细节的理解、可量化的结果</b>。", S_P))

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=ML, rightMargin=MR,
                        topMargin=MT, bottomMargin=MB,
                        title="BEVFormer 模型调试实战复盘 2026-09-29",
                        author="学习笔记")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print("OK ->", OUT)
