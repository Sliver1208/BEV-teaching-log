"""
Markdown → PDF 转换脚本
使用 fpdf2 + 微软雅黑字体支持中文
"""
import re
import os
import sys
from fpdf import FPDF


class ChinesePDF(FPDF):
    def __init__(self):
        super().__init__()
        # fpdf2 v2.5+ 不需要 uni=True
        self.add_font("msyh", "", r"C:\Windows\Fonts\msyh.ttc")
        self.add_font("msyh", "B", r"C:\Windows\Fonts\msyhbd.ttc")
        self.add_font("msyh", "I", r"C:\Windows\Fonts\msyhl.ttc")
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        if self.page_no() > 1:
            self.set_font("msyh", "I", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10, "BEV 深度解析 + 架构师聊天全解析", align="R")
            self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("msyh", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"第 {self.page_no()} 页", align="C")

    def add_title(self, text, level=1):
        self.set_x(self.l_margin)
        self.ln(4)
        w = self.w - self.l_margin - self.r_margin
        if level == 1:
            self.set_font("msyh", "B", 20)
            self.set_text_color(220, 50, 50)
            self.multi_cell(w, 14, text)
            self.set_draw_color(220, 50, 50)
            self.set_line_width(0.5)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(4)
        elif level == 2:
            self.set_font("msyh", "B", 15)
            self.set_text_color(50, 50, 150)
            self.multi_cell(w, 10, text)
            self.ln(2)
        elif level == 3:
            self.set_font("msyh", "B", 12)
            self.set_text_color(80, 80, 80)
            self.multi_cell(w, 8, text)
            self.ln(2)
        elif level == 4:
            self.set_font("msyh", "B", 11)
            self.set_text_color(100, 100, 100)
            self.multi_cell(w, 7, text)
            self.ln(1)

    def add_paragraph(self, text):
        """处理行内 **bold** 和 `code`，用 multi_cell 统一渲染"""
        self.set_font("msyh", "", 10)
        self.set_text_color(30, 30, 30)
        self.set_x(self.l_margin)

        # 先把 **bold** 替换成简单格式，`code` 也替换
        # fpdf2 原生不支持行内样式切换，所以简化处理：
        # 把 **text** 渲染为 【text】，`text` 渲染为 text
        text = re.sub(r'\*\*([^*]+)\*\*', r'【\1】', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)

        # 确保有足够宽度
        w = self.w - self.l_margin - self.r_margin
        if self.get_x() + w > self.w:
            self.set_x(self.l_margin)

        self.multi_cell(w, 6, text)
        self.ln(2)

    def add_code_block(self, code_lines):
        self.set_font("msyh", "", 8.5)
        self.set_text_color(80, 80, 80)
        self.set_fill_color(245, 245, 245)
        self.set_draw_color(200, 200, 200)

        w = self.w - self.l_margin - self.r_margin
        for line in code_lines:
            line_clean = re.sub(r'\x1b\[[0-9;]*m', '', line)
            # 截断过长的行
            if len(line_clean) > 120:
                line_clean = line_clean[:117] + "..."
            self.set_x(self.l_margin)
            self.multi_cell(w, 5, "  " + line_clean[:120], fill=True)
        self.ln(3)

    def add_table(self, header, rows):
        n_cols = len(header)
        # 简单等宽
        col_w = min((self.w - self.l_margin - self.r_margin) / n_cols, 65)

        self.set_font("msyh", "B", 9)
        self.set_fill_color(220, 230, 250)
        self.set_text_color(30, 30, 30)
        x0 = self.get_x()
        for h in header:
            self.cell(col_w, 8, str(h)[:20], border=1, fill=True, align="L")
        self.ln()

        self.set_font("msyh", "", 9)
        for r_idx, row in enumerate(rows):
            if self.get_y() > self.h - 30:
                self.add_page()
                self.set_font("msyh", "", 9)
            fill = r_idx % 2 == 0
            for cell in row:
                cell_text = str(cell).replace("**", "").replace("`", "")[:30]
                self.cell(col_w, 7, cell_text, border=1, fill=fill, align="L")
            self.ln()
        self.ln(3)

    def add_divider(self):
        self.set_draw_color(200, 200, 200)
        self.set_line_width(0.3)
        self.line(10, self.get_y() + 2, 200, self.get_y() + 2)
        self.ln(6)


def parse_markdown(md_path, pdf_path):
    pdf = ChinesePDF()
    pdf.add_page()

    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    in_code_block = False
    code_lines = []
    in_table = False
    table_header = []
    table_rows = []

    while i < len(lines):
        line = lines[i].rstrip()

        # 代码块边界
        if line.startswith("```"):
            if not in_code_block:
                in_code_block = True
                code_lines = []
            else:
                in_code_block = False
                if code_lines:
                    pdf.add_code_block(code_lines)
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        # 空行
        if not line.strip():
            i += 1
            continue

        # 表格
        if line.strip().startswith("|") and "|" in line[1:]:
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if all(set(c) <= {"-", ":", " "} for c in cells):
                i += 1
                continue
            if not in_table:
                in_table = True
                table_header = cells
                table_rows = []
            else:
                table_rows.append(cells)
            i += 1
            continue
        elif in_table and not line.strip().startswith("|"):
            if table_header and table_rows:
                pdf.add_table(table_header, table_rows)
            in_table = False
            table_header = []
            table_rows = []

        # 标题
        if line.startswith("#### "):
            pdf.add_title(line[5:], level=4)
        elif line.startswith("### "):
            pdf.add_title(line[4:], level=3)
        elif line.startswith("## "):
            pdf.add_title(line[3:], level=2)
        elif line.startswith("# "):
            pdf.add_title(line[2:], level=1)
        elif line == "---":
            pdf.add_divider()
        elif line.startswith("> "):
            pdf.set_font("msyh", "I", 9)
            pdf.set_text_color(100, 100, 100)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin, 5, "  " + line[2:])
            pdf.ln(2)
        elif line.startswith("- ") or line.startswith("* "):
            pdf.set_font("msyh", "", 10)
            pdf.set_text_color(30, 30, 30)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin, 6, "  • " + line[2:])
        elif re.match(r"^\d+\.\s", line):
            pdf.set_font("msyh", "", 10)
            pdf.set_text_color(30, 30, 30)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin, 6, "  " + line)
        elif line.startswith("✅") or line.startswith("🔴") or line.startswith("🟡") or line.startswith("🟢"):
            # emoji 可能会有编码问题，替换成简单标记
            clean = line.replace("✅", "[OK] ").replace("🔴", "[!] ").replace("🟡", "[*] ").replace("🟢", "[-] ")
            clean = clean.replace("🚀", "").replace("📌", ">> ").replace("📅", "Date: ").replace("💡", "[Tip] ")
            pdf.add_paragraph(clean)
        else:
            pdf.add_paragraph(line)

        i += 1

    if in_table and table_header and table_rows:
        pdf.add_table(table_header, table_rows)

    pdf.output(pdf_path)
    print(f"PDF 生成成功: {pdf_path}")
    print(f"文件大小: {os.path.getsize(pdf_path) / 1024:.1f} KB")


if __name__ == "__main__":
    # 支持命令行传文件路径，否则用默认文件
    if len(sys.argv) >= 2:
        md_file = sys.argv[1]
        pdf_file = sys.argv[2] if len(sys.argv) >= 3 else os.path.splitext(md_file)[0] + ".pdf"
    else:
        md_file = r"c:\Users\PC\Documents\trae_projects\bev\notes\05_老哥聊天全解析_秋招行动指南.md"
        pdf_file = r"c:\Users\PC\Documents\trae_projects\bev\notes\05_老哥聊天全解析_秋招行动指南.pdf"

    if not os.path.exists(md_file):
        print(f"文件不存在: {md_file}")
        sys.exit(1)

    parse_markdown(md_file, pdf_file)
