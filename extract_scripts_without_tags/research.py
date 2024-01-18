import os

from bs4 import BeautifulSoup

SCRIPT_HTML_PATH = 'script_html/raw_script_html'
filename = '12 Monkeys.txt'
filename = 'Talented Mr. Ripley, The.txt'


filepath = os.path.join(SCRIPT_HTML_PATH, filename)
with open(filepath) as raw_script_file:
    title = next(raw_script_file)
    url = next(raw_script_file)
    html = raw_script_file.read()
    soup = BeautifulSoup(html, 'html5lib')

    for script in soup.find_all('script'):
        script.extract()
    for bold in soup.find_all('b'):
        bold.extract()

    script_text = soup.get_text()
    table_start = script_text.rfind(title)
    script_text = script_text[:table_start]

    lines_without_whitespace = []
    for line in script_text.splitlines():
        if 'This is not' in line:
            print(repr(line.lstrip()))
            print(repr(line))
        if line and not line.isspace():
            lines_without_whitespace.append(line + os.linesep)
    for line in lines_without_whitespace:
        continue
        print(line)
        


    