import os
import time
import pickle
import sys

import torch
import torch.nn as nn
from tqdm import tqdm

import load_data
import tokenize_script
import dataset
import model

DATA_PATH = 'script_files/joined/raw.txt'
TOKENIZED_DATA_PATH = 'script_files/tokenized/single'
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
    dialogue, screen_directions = load_data.get_three_line_dialogue_and_stage_directions(DATA_PATH)
    tokenized_dialogue = tokenizer.texts_to_sequences(dialogue)
    tokenized_screen_directions = tokenizer.texts_to_sequences(screen_directions)

    tokenized_screen_directions = torch.IntTensor(tokenized_screen_directions)
    tokenized_dialogue = torch.IntTensor(tokenized_dialogue)
    torch.save(tokenized_screen_directions, os.path.join(TOKENIZED_DATA_PATH, 'screen_directions.pt'))
    torch.save(tokenized_dialogue, os.path.join(TOKENIZED_DATA_PATH, 'dialogue.pt'))
else:
    tokenized_dialogue = torch.load(os.path.join(TOKENIZED_DATA_PATH, 'dialogue.pt'))
    tokenized_screen_directions = torch.load(os.path.join(TOKENIZED_DATA_PATH, 'screen_directions.pt'))



labeled_data = []
start = time.time()
for line in tokenized_dialogue:
    labeled_data.append((line, 0))
for line in tokenized_screen_directions:
    labeled_data.append((line, 1))

print(labeled_data[0][0])




script_dataset = dataset.ScriptDataset(labeled_data)
train_dataset, test_dataset = torch.utils.data.random_split(script_dataset, [0.7, 0.3])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)

BLOCKS = 0
FILTERS = 32
EMBEDDING_DIM = 200
KERNEL_SIZE = 3
DROPOUT = 0.3
LEARNING_RATE = 1E-3
NUM_EPOCHS = 10


classifier_model = model.Classifier(blocks=BLOCKS, filters=FILTERS, kernel_size=KERNEL_SIZE,
                                    embedding_dim=EMBEDDING_DIM, dropout_rate=DROPOUT, pool_size=KERNEL_SIZE, 
                                    input_shape=[BATCH_SIZE, len(train_dataset[0][0])], num_features=tokenizer.get_vocab_length()).to(device)

sigmoid = nn.Sigmoid()

classifier_model.load_state_dict(torch.load('classification/models/classifier_model.pt'))
criterion = nn.BCELoss()
classifier_model.eval()

iterations = 0
total_guesses = 0
total_correct = 0

running_loss = 0
for epoch in range(1):
    total_train_loss = 0
    for x, y in tqdm(test_loader):
        x = x.to(device)
        x = torch.tensor(x)
        y = y.to(device).float()

        outputs = classifier_model(x)

        loss = criterion(sigmoid(outputs), y) 

        guesses = torch.round(sigmoid(outputs))
        # print(torch.sum(guesses == y).item()/x.size(0))

        iterations += 1
        running_loss += loss.item() * x.size(0)

        total_train_loss += loss.item() * x.size(0)

        # print(tokenizer.sequence_to_text(x[0]), y[0].item(), guesses[0].item())
        total_guesses += x.size(0)
        total_correct += torch.sum(guesses == y).item()
        print(total_correct/total_guesses)


    print(f'loss: {total_train_loss/(epoch + 1)}')
