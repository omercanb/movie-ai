import os
import time
import pickle
import sys

import torch
import torch.nn as nn
from tqdm import tqdm
from matplotlib import pyplot as plt

import load_data
import tokenize_script
import dataset
import model


DATA_PATH = 'classification/dataset/sentence'

TOKENIZED_DATA_PATH = 'classification/dataset/tokenized/sentence/'

TOKENIZER_PATH = 'classification/tokenizer.pkl'

device = 'mps'


if not os.path.exists(TOKENIZER_PATH):
    train = load_data.get_labeled_data(DATA_PATH, 'train')
    train_text, train_labels = zip(*train)
    test = load_data.get_labeled_data(DATA_PATH, 'test')
    test_text, test_labels = zip(*test)

    tokenizer = tokenize_script.Tokenizer()

    tokenizer.fit_on_texts(train_text + test_text)
    with open(TOKENIZER_PATH, 'wb') as tokenizer_pickle:
        pickle.dump(tokenizer.word_index ,tokenizer_pickle)
    print('tokenizer saved')
else:
    tokenizer_word_to_index_dict = pickle.load(open(TOKENIZER_PATH, 'rb'))
    tokenizer = tokenize_script.Tokenizer(word_index=tokenizer_word_to_index_dict)


if not os.path.exists(TOKENIZED_DATA_PATH):
    os.makedirs(TOKENIZED_DATA_PATH)
    os.mkdir(os.path.join(TOKENIZED_DATA_PATH, 'train'))
    os.mkdir(os.path.join(TOKENIZED_DATA_PATH, 'test'))

    train = load_data.get_labeled_data(DATA_PATH, 'train')
    test = load_data.get_labeled_data(DATA_PATH, 'test')

    train_text, train_labels = zip(*train)
    test_text, test_labels = zip(*test)

    train_tokenized = tokenizer.texts_to_sequences(train_text)
    test_tokenized = tokenizer.texts_to_sequences(test_text)

    train_tokenized = torch.IntTensor(train_tokenized)
    test_tokenized = torch.IntTensor(test_tokenized)

    train_labels = torch.IntTensor(train_labels)
    test_labels = torch.IntTensor(test_labels)
    
    torch.save(train_tokenized, os.path.join(TOKENIZED_DATA_PATH, 'train/text.pt'))
    torch.save(train_labels, os.path.join(TOKENIZED_DATA_PATH, 'train/labels.pt'))

    torch.save(test_tokenized, os.path.join(TOKENIZED_DATA_PATH, 'test/text.pt'))
    torch.save(test_labels, os.path.join(TOKENIZED_DATA_PATH, 'test/labels.pt'))
else:
    train_tokenized = torch.load(os.path.join(TOKENIZED_DATA_PATH, 'train/text.pt'))
    train_labels = torch.load(os.path.join(TOKENIZED_DATA_PATH, 'train/labels.pt'))

    test_tokenized = torch.load(os.path.join(TOKENIZED_DATA_PATH, 'test/text.pt'))
    test_labels = torch.load(os.path.join(TOKENIZED_DATA_PATH, 'test/labels.pt'))


BATCH_SIZE = 512
BLOCKS = 4
FILTERS = 32
EMBEDDING_DIM = 200
KERNEL_SIZE = 3
DROPOUT = 0.3
LEARNING_RATE = 1E-3
NUM_EPOCHS = 10

# train_tokenized = train_tokenized[:10000]
# train_labels = train_labels[:10000]

# test_tokenized = test_tokenized[:2000]
# test_labels = test_labels[:2000]

train_labeled = list(zip(train_tokenized, train_labels))
test_labeled = list(zip(test_tokenized, test_labels))
train_dataset = dataset.ScriptDataset(train_labeled)
test_dataset = dataset.ScriptDataset(test_labeled)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)


classifier_model = model.Classifier(blocks=BLOCKS, filters=FILTERS, kernel_size=KERNEL_SIZE,
                                    embedding_dim=EMBEDDING_DIM, dropout_rate=DROPOUT, pool_size=KERNEL_SIZE, 
                                    num_features=tokenizer.get_vocab_length()).to(device)

criterion = nn.BCELoss()
sigmoid = nn.Sigmoid()
optimizer = torch.optim.Adam(classifier_model.parameters(), lr=LEARNING_RATE)

train_loss = []
test_loss = []

highest_precision = 0
best_model = None
best_epoch = 0

for epoch in range(NUM_EPOCHS):
    classifier_model.train()
    running_loss = 0
    
    for i, (x, y) in enumerate(tqdm(train_loader)):
        x = x.to(device)
        y = y.to(device).float()

        outputs = classifier_model(x)

        loss = criterion(sigmoid(outputs), y) 
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()


        guesses = torch.round(sigmoid(outputs))
        TP = torch.sum((guesses == y).logical_and(y == 0))
        FP = torch.sum((guesses != y).logical_and(y == 1))

        running_loss += loss.item() * x.size(0)

        if i % (len(train_loader) // 10) == 0:
            print(f'Precision: {TP/ (FP + TP) if (FP + TP != 0) else 0}')
            print(f'Loss: {loss.item()}')
    train_loss.append(running_loss/len(train_dataset))
    print(f'Train Loss : {train_loss[-1]}')
    
    running_loss = 0
    average_precision = 0

    classifier_model.eval()
    print('Testing')
    TP = 0
    FP = 0
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device).float()
        outputs = classifier_model(x)

        loss = criterion(sigmoid(outputs), y) 
        guesses = torch.round(sigmoid(outputs))
        TP += torch.sum((guesses == y).logical_and(y == 0))
        FP += torch.sum((guesses != y).logical_and(y == 1))
        running_loss += loss.item() * x.size(0)

    test_loss.append(running_loss/len(test_dataset))
    average_precision = TP / (TP + FP)

    print(f'Test Loss: {test_loss[-1]}')
    print(f'Test Precision: {average_precision}')

    if average_precision > highest_precision:
        highest_precision = average_precision
        best_model = classifier_model.state_dict()
        best_epoch = epoch
    

plt.plot(range(NUM_EPOCHS), train_loss, label='Train Loss')
plt.plot(range(NUM_EPOCHS), test_loss, label='Test Loss')
plt.xticks(range(NUM_EPOCHS))
plt.title('Training and Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

print(f'Best Precision: {highest_precision} on epoch {best_epoch}')

plt.savefig('classification/plot.png')
plt.show()

torch.save(best_model, 'classification/models/classifier_model.pt')