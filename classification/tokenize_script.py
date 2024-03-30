import torch

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

    def __init__(self, num_words = TOP_K, word_index = None):
        self.num_words = num_words
        self.word_index = word_index
        if word_index is not None:
            self.set_index_word()

    def fit_on_texts(self, texts: list[str]):
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
        self.set_index_word()


    def set_index_word(self):
        self.index_word = {v:k for k, v in self.word_index.items()}


    def sequence_to_text(self, sequence):
        result = ' '.join([self.index_word.get(int(key), 'UNK') for key in sequence])
        return result[:result.index(self.pad_token)]


    def texts_to_sequences(self, texts, length = MAX_LENGTH):
        return [self.text_to_sequence(text, length) for text in texts]


    def text_to_sequence(self, text, length = MAX_LENGTH):
        sequence = [self.word_index.get(word, self.word_index[Tokenizer.out_of_vocab_token]) for word in Tokenizer.word_tokenize(text)][:length]
        while len(sequence) < length:
            sequence.append(self.word_index[Tokenizer.pad_token])
        return sequence
    

    def word_tokenize(text):
        text = text.lower()
        text = text.translate(Tokenizer.translate_map)
        sequence = text.split(Tokenizer.split)
        return [i for i in sequence if i]
    

    def get_vocab_length(self):
        return len(self.word_index)
        



if __name__ == '__main__':
    DATA_PATH = 'script_files/joined/raw.txt'

    dialogue, screen_directions = load_data.get_dialogue_and_stage_directions(DATA_PATH)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(dialogue + screen_directions)
    print("Tokenizer fitted")

    dialogue, screen_directions = load_data.get_three_line_dialogue_and_stage_directions(DATA_PATH)
    print("Data loaded")

    tokenized_dialogue = tokenizer.texts_to_sequences(dialogue[:100])
    tokenized_screen_directions = tokenizer.texts_to_sequences(screen_directions[:100])
    print('SHOOOOWm')
    print(type(tokenized_dialogue), type(tokenized_dialogue[0]))
    print(type(tokenized_dialogue[0][0]))
    x = torch.IntTensor(tokenized_dialogue)
    torch.save(x, 'script_files/tokenized/dialogue.pt')




