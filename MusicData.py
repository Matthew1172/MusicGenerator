from torch.nn.utils.rnn import pad_sequence
import os
import torch
import pickle
from Dictionary import *
from torch.utils.data import Dataset, DataLoader

def get_valid_loader(dataset, batch_size, num_workers=0, shuffle=True, pin_memory=True):
    pad_idx = 0
    loader = DataLoader(dataset, batch_size = batch_size, num_workers = num_workers,
                        shuffle=shuffle,
                       pin_memory=pin_memory, collate_fn = MyCollate(pad_idx=pad_idx))
    return loader

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    # __call__: a default method
    ##   First the obj is created using MyCollate(pad_idx) in data loader
    ##   Then if obj(batch) is called -> __call__ runs by default
    def __call__(self, batch):
        # get all source indexed sentences of the batch
        source = [item['song'] for item in batch]
        # pad them using pad_sequence method from pytorch.
        source = pad_sequence(source, batch_first=False, padding_value=self.pad_idx)

        return source

class ABCMusicDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, my_21_abc_file, transform=None):
        """
        Args:
            my_21_abc_file (string): Path to the binary file with each element being an array of my music21 abc format. ex.
            [["Clef G", "Time 4 4", ... ],
            ["Clef G", "Time 2 4", ... ],
            ["Clef G", "Time 3 4", ... ],
            ...]
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.my_abc = self.tokenize_from_bin(my_21_abc_file)
        self.dic = Dictionary()
        self.createDictionary(self.my_abc)
        self.dataset = self.tokenizeFileContent(self.my_abc)
        self.transform = transform

    def __len__(self):
        return len(self.my_abc)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        song = self.my_abc[idx]
        sample = {'song': torch.tensor([self.dic.word2idx[note] for note in song])}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def createDictionary(self, mySongFormatCombined):
        self.dic.add_word("<PAD>")
        for ps in mySongFormatCombined:
            for ele in ps:
                self.dic.add_word(ele)

    def tokenize_from_bin(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        with open(path, 'rb') as f:
            pretty_info = pickle.load(f)
        return pretty_info

    def tokenizeFileContent(self, mySongFormat):
        # Tokenize file content
        idss = []
        for s in mySongFormat:
            ids = []
            for n in s:
                ids.append(self.dic.word2idx[n])
            idss.append(torch.tensor(ids, dtype=torch.int64))
        ids = torch.cat(idss)
        return ids
