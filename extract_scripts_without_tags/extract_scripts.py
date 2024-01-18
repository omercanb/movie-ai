import os

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
        open_parantheses = False
        for line in script_text.splitlines():
            if line and not line.isspace():
                if open_parantheses: #To not delete any more than two lines
                    open_parantheses = False 
                    continue
                if line.lstrip()[0] == '(':
                    if line.rstrip()[-1] != ')':
                        open_parantheses = True
                    continue
                lines_without_whitespace.append(line + os.linesep)

        lines_without_characters = []
        for line in lines_without_whitespace:
            if '(' in line and ')' in line:
                line_without_parantheses = line[:line.find('(')] + line[line.find(')'):]
                if line_without_parantheses.upper() == line_without_parantheses:
                    continue
            lines_without_characters.append(line)
        
        with open(os.path.join(EXTRACTED_SCRIPT_PATH, filename), 'w') as f:
            f.writelines([title, url])
            f.writelines(lines_without_characters[:-1])
            f.write(lines_without_characters[-1].rstrip())
        number_of_scripts_extracted  += 1
        print(f'Extracted {title.strip()}, {number_of_scripts_extracted}/{NUMBER_OF_TOTAL_SCRIPTS}')


