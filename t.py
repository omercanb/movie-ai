def word_tokenize(text):
    filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    split = ' '
    translate_dict = {c: split for c in filters}
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)
    sequence = text.split(split)
    return [i for i in sequence if i]


s = 'This is a sample, don\'t sentence.'
print(word_tokenize(s))