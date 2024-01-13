import os

SCRIPT_PATH = 'script_files/raw_script_html'
REMOVED_SCRIPT_INFO_PATH = 'get_script_html/removed_script_info.txt'

if not os.path.exists(REMOVED_SCRIPT_INFO_PATH):
    open(REMOVED_SCRIPT_INFO_PATH, 'w')

min_size_paths = []
min_size = 1e100
for filename in os.listdir(SCRIPT_PATH):
    filepath = os.path.join(SCRIPT_PATH, filename)
    with open(filepath) as f:
        size = len(f.readlines())
        if size < min_size:
            min_size_paths.clear()
            min_size = size
            min_size_paths.append(filepath)
        elif min_size == size:
            min_size_paths.append(filepath)

with open(min_size_paths[-1]) as f:
    print(f.read())

print(str(len(min_size_paths)) + ' files of this length.')
print(min_size_paths)
clear = input('Clear files of this length ?: ')
if clear == 'y':
    removed_script_info_file = open(REMOVED_SCRIPT_INFO_PATH, 'a')
    for path in min_size_paths:
        with open(path) as f:
            title = next(f)
            url = next(f)
        os.remove(path)
        removed_script_info_file.write(title)
        removed_script_info_file.write(url + os.linesep)
    removed_script_info_file.write(os.linesep)
    removed_script_info_file.close()

