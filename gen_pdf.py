# -*- coding: utf-8 -*-
"""
將 Markdown 學習報告轉換為 PDF（支持中文、高亮、圖片）
使用 markdown -> HTML -> xhtml2pdf 鏈路
"""
import os
import markdown
from xhtml2pdf import pisa

# 路徑配置
BASE_DIR = r"c:\Users\PC\Documents\trae_projects\bev"
MD_FILE = os.path.join(BASE_DIR, "notes", "12_电机方向实习经历包装指南_20260928.md")
PDF_FILE = os.path.join(BASE_DIR, "notes", "12_电机方向实习经历包装指南_20260928.pdf")

# Windows 中文字體路徑（黑體，已複製到項目目錄）
FONT_PATH = os.path.join(BASE_DIR, "simhei.ttf")


def link_callback(uri, rel):
    """處理 HTML 中的資源路徑（字體 + 圖片）"""
    # 字體
    if uri == "simhei.ttf":
        return FONT_PATH
    # 圖片：md 裡用 ../vis_results/xxx.png，需要解析到項目根目錄
    if "vis_results" in uri or uri.endswith((".png", ".jpg", ".jpeg")):
        # 去掉 ../ 前綴，從項目根目錄查找
        clean = uri.lstrip("./")
        candidate = os.path.join(BASE_DIR, clean)
        if os.path.exists(candidate):
            return candidate
        # 再試 vis_results 直接拼接
        candidate2 = os.path.join(BASE_DIR, "vis_results", os.path.basename(uri))
        if os.path.exists(candidate2):
            return candidate2
    return uri


def md_to_pdf(md_path, pdf_path):
    # 1. 讀取 md 並轉為 HTML
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "codehilite", "toc"],
    )

    # 2. 構造完整 HTML，注入中文字體 + 高亮 + 圖片樣式
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
@font-face {{
    font-family: 'SimHei';
    src: url('simhei.ttf');
}}
@page {{
    size: a4;
    margin: 2cm 1.8cm 2cm 1.8cm;
}}
body {{
    font-family: 'SimHei', sans-serif;
    font-size: 11pt;
    line-height: 2.0;
    color: #222;
    word-wrap: break-word;
    word-break: break-word;
}}
h1 {{
    font-size: 20pt;
    color: #1a5276;
    border-bottom: 2px solid #1a5276;
    padding-bottom: 8px;
    margin-bottom: 16px;
    page-break-after: avoid;
}}
h2 {{
    font-size: 15pt;
    color: #21618c;
    margin-top: 26px;
    margin-bottom: 12px;
    border-left: 4px solid #21618c;
    padding-left: 10px;
    page-break-after: avoid;
}}
h3 {{
    font-size: 13pt;
    color: #2874a6;
    margin-top: 18px;
    margin-bottom: 10px;
    page-break-after: avoid;
}}
p {{
    margin: 10px 0;
    text-align: justify;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0;
    font-size: 9.5pt;
    table-layout: fixed;
    word-wrap: break-word;
    page-break-inside: avoid;
}}
th {{
    background-color: #21618c;
    color: white;
    padding: 8px 6px;
    text-align: left;
    border: 1px solid #1a5276;
    word-wrap: break-word;
}}
td {{
    padding: 7px 6px;
    border: 1px solid #bbb;
    vertical-align: top;
    word-wrap: break-word;
    overflow-wrap: break-word;
}}
tr:nth-child(even) {{
    background-color: #f2f7fb;
}}
mark {{
    background-color: #fff59d;
    color: #222;
    padding: 1px 4px;
    border-radius: 2px;
    line-height: 1.6;
}}
code {{
    background-color: #f4f4f4;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 9.5pt;
    color: #c0392b;
    font-family: 'SimHei', monospace;
    word-wrap: break-word;
}}
pre {{
    background-color: #f4f4f4;
    padding: 10px 12px;
    border-left: 3px solid #21618c;
    margin: 12px 0;
    font-size: 8.5pt;
    line-height: 1.6;
    font-family: 'SimHei', monospace;
    white-space: pre-wrap;
    word-wrap: break-word;
    overflow-wrap: break-word;
    page-break-inside: avoid;
}}
blockquote {{
    border-left: 4px solid #3498db;
    padding: 8px 16px;
    margin: 12px 0;
    color: #555;
    background-color: #eaf2f8;
    page-break-inside: avoid;
}}
hr {{
    border: none;
    border-top: 1px solid #ccc;
    margin: 18px 0;
}}
ul, ol {{
    margin: 8px 0;
    padding-left: 24px;
}}
li {{
    margin: 5px 0;
}}
img {{
    max-width: 90%;
    height: auto;
    display: block;
    margin: 12px auto;
    border: 1px solid #ddd;
    padding: 4px;
    page-break-inside: avoid;
}}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

    # 3. HTML -> PDF
    with open(pdf_path, "wb") as f:
        result = pisa.CreatePDF(
            html.encode("utf-8"),
            dest=f,
            link_callback=link_callback,
            encoding="utf-8",
        )

    if result.err:
        print(f"[錯誤] PDF 生成失敗：{result.err}")
        return False
    else:
        size_mb = os.path.getsize(pdf_path) / 1024 / 1024
        print(f"[成功] PDF 已生成：{pdf_path}（{size_mb:.2f} MB）")
        return True


if __name__ == "__main__":
    md_to_pdf(MD_FILE, PDF_FILE)
