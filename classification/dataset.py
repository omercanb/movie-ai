from torch.utils.data import Dataset

class ScriptDataset(Dataset):
    def __init__(self, labeled_data):
        self.data = labeled_data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]