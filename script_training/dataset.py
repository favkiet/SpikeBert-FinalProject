from torch.utils.data import Dataset
import random
from tqdm import tqdm
import pandas as pd

class TensorDataset(Dataset):
    def __init__(self, data: str):
        super(TensorDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        embedding = self.data[index][0]
        label = int(self.data[index][1])
        return embedding, label

class RateDataset(Dataset):
    def __init__(self, data: str):
        super(RateDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        rate_code = self.data[index][0]
        label = int(self.data[index][1])
        return rate_code, label

class TxtDataset(Dataset):
    def __init__(self, data_path: str):
        super(TxtDataset, self).__init__()
        with open(data_path) as fin:
            self.lines = fin.readlines()
        
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index: int):
        line = self.lines[index]
        line = line.strip()
        temp = line.split('\t')
        # print(temp)
        sentence = temp[0]
        label = int(temp[1])
        return sentence, label

class TextDataset(Dataset):
    def __init__(self, raw_dataset):
        super(TextDataset, self).__init__()
        self.dataset = raw_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index]["text"]
    
class ChnWikiDataset(Dataset):
    def __init__(self, data_path: str):
        super(ChnWikiDataset, self).__init__()
        with open(data_path) as fin:
            self.lines = fin.readlines()
        
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index: int):
        line = self.lines[index]
        line = line.strip()
        return line

class SentencePairDataset(Dataset):
    def __init__(self, data_path: str):
        super(SentencePairDataset, self).__init__()
        with open(data_path) as fin:
            self.lines = fin.readlines()
        
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index: int):
        line = self.lines[index]
        line = line.strip()
        temp = line.split('\t')
        # print(temp)
        sentence_pair = [temp[0], temp[1]]
        label = float(temp[2])
        return sentence_pair, label

class CSVDataset(Dataset):
    def __init__(self, data_path: str, sentence_col: str = 'sentence', label_col: str = 'sentiment'):
        super(CSVDataset, self).__init__()
        self.df = pd.read_csv(data_path)
        self.sentence_col = sentence_col
        self.label_col = label_col
        
        # Tạo mapping từ text label sang số
        self.label_mapping = self._create_label_mapping()
        
    def _create_label_mapping(self):
        """Tạo mapping từ text label sang số"""
        unique_labels = self.df[self.label_col].unique()
        label_mapping = {}
        
        # Xử lý các trường hợp label khác nhau
        for i, label in enumerate(sorted(unique_labels)):
            if isinstance(label, str):
                label_lower = label.lower()
                if 'negative' in label_lower:
                    label_mapping[label] = 0
                elif 'positive' in label_lower:
                    label_mapping[label] = 2 if len(unique_labels) == 3 else 1
                elif 'neutral' in label_lower or 'neu' in label_lower:
                    label_mapping[label] = 1
                else:
                    label_mapping[label] = i
            else:
                label_mapping[label] = int(label)
                
        return label_mapping
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        sentence = self.df.iloc[index][self.sentence_col]
        label_text = self.df.iloc[index][self.label_col]
        label = self.label_mapping[label_text]
        return sentence, label