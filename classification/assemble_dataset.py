import os

import torch
import nltk

def save_ordered_train_test_split(ordered_data_path, save_path, split):
    files = os.listdir(ordered_data_path)
    train, test = torch.utils.data.random_split(files, split)
    with open(os.path.join(save_path, 'train.txt'), 'w') as f:
        for fname in train:
            f.write(open(os.path.join(ordered_data_path, fname)).read())
    with open(os.path.join(save_path, 'test.txt'), 'w') as f:
        for fname in test:
            f.write(open(os.path.join(ordered_data_path, fname)).read())
    return train, test


def save_ordered_sentence_train_test_split(ordered_data_path, save_path, split):
    files = os.listdir(ordered_data_path)
    train, test = torch.utils.data.random_split(files, split)
    train_sentences = []
    test_sentences = []
    for fname in train:
        train_sentences.extend(nltk.tokenize.sent_tokenize(open(os.path.join(ordered_data_path, fname)).read()))
    for fname in test:
        test_sentences.extend(nltk.tokenize.sent_tokenize(open(os.path.join(ordered_data_path, fname)).read()))
    print('tokenized')

    for sentences in train_sentences, test_sentences:
        label = sentences[0][0]
        for i in range(len(sentences)):
            sentences[i] = sentences[i].replace('\n0', '').replace('\n1', '')
            if sentences[i][0] != "0" and sentences[i][0] != "1":
                sentences[i] = label + ' ' + sentences[i]
            else:
                label = sentences[i][0]

    print(len(train_sentences))
    print(len(test_sentences))
        
    with open(os.path.join(save_path, 'train.txt'), 'w') as f:
        f.writelines('\n'.join(train_sentences))
    with open(os.path.join(save_path, 'test.txt'), 'w') as f:
        f.writelines('\n'.join(test_sentences))

    return train, test


def test_ratios():
    ordered_data_path = 'script_files/ordered_dialogue_and_screen_directions'
    train, test = save_ordered_train_test_split('classification/dataset/line', [0.7, 0.3])
    train_dialogue_screen_direction_split = [0,0]
    test_dialogue_screen_direction_split = [0,0]
    for fname in train:
        with open(os.path.join(ordered_data_path, fname)) as f:
            for line in f.readlines():
                label = int(line[0])
                train_dialogue_screen_direction_split[label] += 1

    for fname in test:
        with open(os.path.join(ordered_data_path, fname)) as f:
            for line in f.readlines():
                label = int(line[0])
                test_dialogue_screen_direction_split[label] += 1
    print(f'Train: {train_dialogue_screen_direction_split}')
    print(f'Test: {test_dialogue_screen_direction_split}')
    print(train_dialogue_screen_direction_split[0]/train_dialogue_screen_direction_split[1])
    print(test_dialogue_screen_direction_split[0]/test_dialogue_screen_direction_split[1])
    # Random split gives the same ratio to 0.1

if __name__ == "__main__":
    ordered_data_path = 'script_files/ordered_dialogue_and_screen_directions'
    train, test = save_ordered_sentence_train_test_split(ordered_data_path, 'classification/dataset/sentence' , [0.7, 0.3])



                