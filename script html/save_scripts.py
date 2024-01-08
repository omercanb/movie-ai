import os

import scripts_from_web

save_path = 'scripts'
checkpoint_path = 'checkpoint.txt'


def get_movie_name(url):
    name_start = url.find('scripts') + len('scripts') + 1
    name_end = url.find('.html')
    name = url[name_start:name_end]
    return name


with open(checkpoint_path) as f:
    checkpoint = f.read()

all_movie_page_urls = scripts_from_web.get_all_movie_page_urls()
movie_page_urls = all_movie_page_urls[all_movie_page_urls.index(checkpoint) + 1:]
urls = (scripts_from_web.get_script_url(movie_page_url) for movie_page_url in movie_page_urls)


for url, movie_page_url in zip(urls, movie_page_urls):
    if url == None:
        continue
    name = get_movie_name(url)
    file_name = name + '.txt'
    #example url: https://imsdb.com/scripts/Joker.html
    print('Saving ' + name)
    try:
        with open(os.path.join(save_path, file_name), 'w') as f:
            f.write(url)
            f.write(os.linesep)
            f.write(str(scripts_from_web.get_script_tag(url)))

    except Exception as e:
        with open(checkpoint_path, 'w') as f:
            f.write(movie_page_url)
        print('Exception occured, stopping.')
        print(e)
        break

with open(checkpoint_path, 'w') as f:
    f.write('All scripts saved!')