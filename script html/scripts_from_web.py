import requests

from bs4 import BeautifulSoup


imsdb_url = 'https://imsdb.com'


def get_all_movie_page_urls():
    url = 'https://imsdb.com/all-scripts.html'
    soup =  BeautifulSoup(requests.get(url).content, 'html5lib')
    tds = soup.select('td')
    td_of_movie_page_hrefs = max(tds, key = lambda x: len(x.select('*', recursive=False)))
    movie_anchors = td_of_movie_page_hrefs.select('a')
    movie_hrefs = [a['href'] for a in movie_anchors]
    movie_page_urls = [imsdb_url + href for href in movie_hrefs]
    return movie_page_urls


def get_script_url(movie_page_url):
    soup = BeautifulSoup(requests.get(movie_page_url).content, 'html5lib')
    main_info_table = soup.find(class_='script-details')
    if main_info_table is None:
        return None
    script_anchor = main_info_table.select('a')[-1]
    anchor_text = script_anchor.text.strip()
    if not anchor_text.startswith('Read') and not anchor_text.endswith('Script'):
        return None
    else:
        script_url = imsdb_url + script_anchor['href']
        return script_url


def get_script_tag(url):
    soup =  BeautifulSoup(requests.get(url).content, 'html5lib')
    tag_with_script = soup.find(class_='scrtext')
    return tag_with_script