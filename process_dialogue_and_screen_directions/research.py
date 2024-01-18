import os
from collections import defaultdict

DIALOGUE_AND_SCREEN_DIRECTIONS_PATH = 'script_files/dialogue_and_screen_directions'
PUNCTUATION = '\'\",.!?'
dictionary = defaultdict(lambda:0)
for i, filename in enumerate(os.listdir(DIALOGUE_AND_SCREEN_DIRECTIONS_PATH)):
    print(i + 1)
    filepath = os.path.join(DIALOGUE_AND_SCREEN_DIRECTIONS_PATH, filename)
    with open(filepath) as f:
        for line in f.readlines():
            letters = []
            for char in line:
                if char.isalpha():
                    letters.append(char)
                else:
                    word = ''.join(letters).lower()
                    dictionary[word] += 1
                    letters.clear()

    

print(sorted(dictionary.items(), key=lambda x:x[1], reverse=True))