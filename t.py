import torch

guesses = torch.tensor([0,0,1,1])
y = torch.tensor([0,1,0,1])

print(guesses != y)
print(y == 1)
print((guesses != y).logical_and((y == 1)))