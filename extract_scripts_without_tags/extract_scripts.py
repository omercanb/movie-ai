import os
import random

from bs4 import BeautifulSoup


SCRIPT_HTML_PATH = 'script_files/raw_script_html'
EXTRACTED_SCRIPT_PATH = 'script_files/scripts_without_tags'

NUMBER_OF_TOTAL_SCRIPTS = len(os.listdir(SCRIPT_HTML_PATH))

if not os.path.exists(EXTRACTED_SCRIPT_PATH):
    os.mkdir(EXTRACTED_SCRIPT_PATH)

number_of_scripts_extracted = 0

for filename in os.listdir(SCRIPT_HTML_PATH):
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
            if line and not line.isspace():
                lines_without_whitespace.append(line + os.linesep)
        
        with open(os.path.join(EXTRACTED_SCRIPT_PATH, filename), 'w') as f:
            f.writelines([title, url])
            f.writelines(lines_without_whitespace[:-1])
            f.write(lines_without_whitespace[-1].rstrip())
        number_of_scripts_extracted  += 1
        print(f'Extracted {title.strip()}, {number_of_scripts_extracted}/{NUMBER_OF_TOTAL_SCRIPTS}')


