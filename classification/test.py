import os
import time
import pickle
import sys
import random

import torch
import torch.nn as nn
from tqdm import tqdm

import load_data
import tokenize_script
import dataset
import model

torch.mps.empty_cache()

DATA_PATH = 'script_files/joined/raw.txt'
TOKENIZED_SINGLE_DATA_PATH = 'script_files/tokenized/single'
TOKENIZED_TRIPLE_DATA_PATH = 'script_files/tokenized/triple'
TEST_DATA_PATH = 'script_files/tokenized/test'
TOKENIZED_DATA_PATH = TEST_DATA_PATH

TOKENIZER_PATH = 'classification/tokenizer.pkl'
BATCH_SIZE = 512

device = 'mps'


if not os.path.exists(TOKENIZER_PATH):
    dialogue, screen_directions = load_data.get_dialogue_and_stage_directions(DATA_PATH)
    tokenizer = tokenize_script.Tokenizer()
    tokenizer.fit_on_texts(dialogue + screen_directions)
    with open(TOKENIZER_PATH, 'wb') as tokenizer_pickle:
        pickle.dump(tokenizer.word_index ,tokenizer_pickle)
    print('tokenizer saved')
else:
    tokenizer_word_to_index_dict = pickle.load(open(TOKENIZER_PATH, 'rb'))
    tokenizer = tokenize_script.Tokenizer(word_index=tokenizer_word_to_index_dict)
    
    
if not os.path.exists(TOKENIZED_DATA_PATH):
    os.mkdir(TOKENIZED_DATA_PATH)
    dialogue, screen_directions = load_data.get_dialogue_and_stage_directions(DATA_PATH)
    labeled_data = []
    labeled_data.extend((line, 0) for line in dialogue)
    labeled_data.extend((line, 1) for line in screen_directions)
    random.shuffle(labeled_data)

    three_line_text = []
    three_line_labels = []
    for i in range(len(labeled_data) - 2):
        three_line_text.append(' '.join((labeled_data[i][0],
                                labeled_data[i+1][0],
                                labeled_data[i+2][0])))
        three_line_labels.append((labeled_data[i][1],
                                labeled_data[i+1][1],
                                labeled_data[i+2][1]))
    
    tokenized_text = [tokenizer.text_to_sequence(l) for l in three_line_text]
    tokenized_text = torch.IntTensor(tokenized_text)
    three_line_labels = torch.IntTensor(three_line_labels)
    torch.save(tokenized_text, os.path.join(TOKENIZED_DATA_PATH, 'data.pt'))
    torch.save(three_line_labels, os.path.join(TOKENIZED_DATA_PATH, 'labels.pt'))



    # print('Dialogue and screen directions loaded')
    # tokenized_dialogue = tokenizer.texts_to_sequences(dialogue)
    # tokenized_screen_directions = tokenizer.texts_to_sequences(screen_directions)
    # print('Data tokenized')
    # tokenized_screen_directions = torch.IntTensor(tokenized_screen_directions)
    # torch.save(tokenized_screen_directions, os.path.join(TOKENIZED_DATA_PATH, 'data.pt'))
    print('Saved')

else:
    tokenized_text = torch.load(os.path.join(TOKENIZED_DATA_PATH, 'data.pt'))
    three_line_labels = torch.load(os.path.join(TOKENIZED_DATA_PATH, 'labels.pt'))

labeled_data = []
labeled_data.extend((text, labels) for text, labels in zip(tokenized_text, three_line_labels))
labeled_data = labeled_data[:10_000]
three_line_labels = three_line_labels[:10_000]



# train_dataset, test_dataset = torch.utils.data.random_split(script_dataset, [0.7, 0.3])
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)

script_dataset = dataset.ScriptDataset(labeled_data)
test_loader = torch.utils.data.DataLoader(script_dataset, batch_size = BATCH_SIZE, shuffle = False)

BLOCKS = 0
FILTERS = 32
EMBEDDING_DIM = 200
KERNEL_SIZE = 3
DROPOUT = 0.3
LEARNING_RATE = 1E-3
NUM_EPOCHS = 10


classifier_model = model.Classifier(blocks=BLOCKS, filters=FILTERS, kernel_size=KERNEL_SIZE,
                                    embedding_dim=EMBEDDING_DIM, dropout_rate=DROPOUT, pool_size=KERNEL_SIZE, 
                                    input_shape=[BATCH_SIZE, len(script_dataset[0][0])], num_features=tokenizer.get_vocab_length()).to(device)

sigmoid = nn.Sigmoid()

classifier_model.load_state_dict(torch.load('classification/models/classifier_model.pt'))
criterion = nn.BCELoss()
classifier_model.eval()

# guess_array = [0] * (len(script_dataset) + 2)
guess_array = torch.zeros(len(script_dataset) + 2 + len(script_dataset) % BATCH_SIZE)
# guess_array = torch.zeros(len(script_dataset))

iterations = 0
total_guesses = 0
total_correct = 0

running_loss = 0

total_train_loss = 0
for x, three_y in tqdm(test_loader):
    x = x.to(device)
    x = torch.tensor(x)
    three_y = three_y.to(device).float()
    y = torch.mean(three_y, 1).round()

    outputs = classifier_model(x)

    loss = criterion(sigmoid(outputs), y) 

    # guesses = torch.round(sigmoid(outputs))
    guesses = sigmoid(outputs)

    # print(torch.sum(guesses == y).item()/x.size(0))

    length = guess_array[BATCH_SIZE * iterations : BATCH_SIZE * (iterations + 1)].size(0)
    expanded_guess = torch.zeros(BATCH_SIZE)
    expanded_guess[:guesses.size(0)] = guesses.to('cpu')
    guess_array[BATCH_SIZE * iterations : BATCH_SIZE * (iterations + 1)] += expanded_guess
    guess_array[BATCH_SIZE * iterations + 1 : BATCH_SIZE * (iterations + 1) + 1] += expanded_guess
    guess_array[BATCH_SIZE * iterations + 2 : BATCH_SIZE * (iterations + 1) + 2] += expanded_guess

    iterations += 1
    running_loss += loss.item() * x.size(0)

print(running_loss)
print(running_loss/len(script_dataset))

guess_array = guess_array[:len(script_dataset)]
guess_array /= 3
guess_array = torch.round(guess_array)
labels = three_line_labels[:,:1].flatten()
print(labels)
print(guess_array)
print(torch.sum(guess_array == labels)/guess_array.size(0))
