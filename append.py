with open('generate_pdf.py', 'r', encoding='utf-8') as f:
    text = f.read()

with open('missing_text.txt', 'r', encoding='utf-8') as f:
    missing = f.read()

new_text = "missing_text = \"\"\"\\\n" + missing + "\"\"\"\n\n" + text

new_text = new_text.replace("pdf.output(", "pdf.add_page()\npdf.multi_cell(0, 6, txt=missing_text)\n\npdf.output(")

with open('generate_pdf.py', 'w', encoding='utf-8') as f:
    f.write(new_text)
