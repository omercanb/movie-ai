import load_dataset

import numpy as np


def get_num_words_per_sample(sample_texts):
    num_words = [len(s.split()) for s in sample_texts]
    return np.median(num_words)

if __name__ == '__main__':
    dialogue, screen_directions = load_dataset.get_three_sentence_dialogue_and_screen_directions()
    words_per_sample = get_num_words_per_sample(dialogue + screen_directions)
    number_of_samples = len(dialogue) + len(screen_directions)
    print(words_per_sample)
    print(number_of_samples)
    print(number_of_samples / words_per_sample)

