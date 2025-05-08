# استبدال السهم غير المدعوم "→" بسهم نصي بسيط "->"
final_text_safe = final_clean_text.replace("→", "->")

# إنشاء PDF آمن بالكامل
pdf = FPDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.set_font("Arial", size=12)

# إضافة النص المنظف
for line in final_text_safe.strip().split('\n'):
    pdf.multi_cell(0, 10, line)

# حفظ التقرير
pdf_path = "/mnt/data/Book_Data_Project_Report.pdf"
pdf.output(pdf_path)


