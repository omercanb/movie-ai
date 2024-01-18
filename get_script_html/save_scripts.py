import os

import requests
from bs4 import BeautifulSoup


ALL_SCRIPTS_URL = 'https://imsdb.com/all-scripts.html'
BASE_SCRIPT_URL = 'https://imsdb.com/scripts/'

SCRIPT_PATH = 'script_files/raw_script_html'
CHECKPOINT_PATH = 'get_script_html/checkpoint.txt'
EMPTY_SCRIPT_MOVIES = 'get_script_html/empty_script_movies.txt'

EMPTY_SCRIPT_HTML = requests.get(BASE_SCRIPT_URL + 'invalid-title.html').content


if not os.path.exists(SCRIPT_PATH):
    os.mkdir(SCRIPT_PATH)
if not os.path.exists(CHECKPOINT_PATH):
    open(CHECKPOINT_PATH, 'x')
if not os.path.exists(EMPTY_SCRIPT_MOVIES):
    open(EMPTY_SCRIPT_MOVIES, 'x')

with open(CHECKPOINT_PATH) as checkpoint_file:
    checkpoint = checkpoint_file.read()


soup =  BeautifulSoup(requests.get(ALL_SCRIPTS_URL).content, 'html5lib')
#The td tag with the most direct children contains all movie page anchors
td_of_movie_page_anchors = max(soup.select('td'), key = lambda x: len(x.select('*', recursive=False)))
movie_anchors = td_of_movie_page_anchors.find_all('a')
titles = [a.text for a in movie_anchors]
script_urls = [BASE_SCRIPT_URL + title.replace(' ', '-').replace(':', '') + '.html' for title in titles]

test_empty_url = BASE_SCRIPT_URL + 'other-invalid-title.html'
script_urls.insert(0, test_empty_url)
titles.insert(0, 'Test Empty Script Movie')

if checkpoint == '':
    checkpoint_index = 0
else:
    checkpoint_index = titles.index(checkpoint)
titles = titles[checkpoint_index:]
script_urls = script_urls[checkpoint_index:]

number_of_scripts_saved = checkpoint_index
NUMBER_OF_TOTAL_SCRIPTS = len(script_urls) + number_of_scripts_saved


for title, script_url in zip(titles, script_urls):
    try:
        html = requests.get(script_url).content
        if html == EMPTY_SCRIPT_HTML:
            with open(EMPTY_SCRIPT_MOVIES, 'a') as f:
                f.write(title + os.linesep)
                continue
        soup = BeautifulSoup(html, 'html5lib')
        tag_with_script = soup.find(class_='scrtext')
        with open(os.path.join(SCRIPT_PATH, title) + '.txt', 'w') as f:
            f.write(title + os.linesep)
            f.write(script_url + os.linesep)
            f.write(str(tag_with_script))
        number_of_scripts_saved += 1
        print(f'Saved {title}, {number_of_scripts_saved}/{NUMBER_OF_TOTAL_SCRIPTS}')
    except Exception as e:
        print(e)
        print('Movie that exception occured on: ' + title)
        with open(CHECKPOINT_PATH, 'w') as f:
            f.write(title)
        print('Movie saved to checkpoint')
        break
    except KeyboardInterrupt:
        print()
        with open(CHECKPOINT_PATH, 'w') as f:
            f.write(title)
        print('Movie saved to checkpoint')
        break