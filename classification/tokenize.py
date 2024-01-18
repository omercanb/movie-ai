import load_data

# nltk word tokenize is too slow


TOP_K = 20_000
MAX_LENGTH = 144


class Tokenizer():
    filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    split = ' '
    pad_token = '[PAD]'
    out_of_vocab_token = '[OOV]'
    translate_map = str.maketrans({c: ' ' for c in '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'})

    def __init__(self, num_words):
        self.num_words = num_words


    def fit_to_texts(self, texts: list[str]):
        occurances = dict()
        for text in texts:
            for word in Tokenizer.word_tokenize(text):
                if word in occurances:
                    occurances[word] += 1
                else:
                    occurances[word] = 1
        word_counts = list(occurances.items())
        word_counts.sort(key=lambda x: x[1], reverse = True)
        word_counts = word_counts[:self.num_words]

        sorted_vocab = [Tokenizer.pad_token, Tokenizer.out_of_vocab_token]
        sorted_vocab.extend(word_count[0] for word_count in word_counts)

        self.word_index = dict(zip(sorted_vocab, list(range(len(sorted_vocab)))))


    def texts_to_sequences(self, texts, length = MAX_LENGTH):
        return [self.text_to_sequence(text, length) for text in texts]


    def text_to_sequence(self, text, length = MAX_LENGTH):
        sequence = [self.word_index.get(word, Tokenizer.out_of_vocab_token) for word in Tokenizer.word_tokenize(text)][:length]
        while len(sequence) < length:
            sequence.append(self.word_index[Tokenizer.pad_token])
        return sequence
    

    def word_tokenize(text):
        text = text.lower()
        text = text.translate(Tokenizer.translate_map)
        sequence = text.split(Tokenizer.split)
        return [i for i in sequence if i]



if __name__ == '__main__':
    dialogue, screen_directions = load_data.get_dialogue_and_stage_directions()
    tokenizer = Tokenizer(num_words = TOP_K)
    tokenizer.fit_to_texts(dialogue + screen_directions)
    dialogue, screen_directions = load_data.get_three_line_dialogue_and_stage_directions()
    print(dialogue[0])
    print(tokenizer.text_to_sequence(dialogue[0]))
    print('DONE')
    print(dialogue[:3])
    print(tokenizer.texts_to_sequences(dialogue[:3]))

