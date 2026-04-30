import torch
from torch.utils.data import Dataset
import numpy as np
from datasets.utils.utils import trim_long_sequences, remove_short_sequences_numpy


class GRUDDataset(Dataset):
    def __init__(self, data, labels, ids, lengths, static_data):
        self.data = data
        self.labels = labels
        self.ids = ids
        self.lengths = lengths
        self.static_data = static_data
        self.dtype = torch.float32

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        patient_id = self.ids[idx]
        length = self.lengths[idx]
        static_data = self.static_data[idx]
        label = torch.tensor([label], dtype=self.dtype)
        sample = torch.tensor(np.array(sample), dtype=self.dtype)
        length = torch.tensor([length])
        static_data = torch.tensor(static_data, dtype=self.dtype)
        patient_id = torch.tensor([patient_id], dtype=self.dtype)
        return sample, label, patient_id, length, static_data


class Torch_Dataset(Dataset):
    def __init__(self, file_path, labels_file_path, ids_file_path, static_file_path):
        self.data = np.load(file_path, allow_pickle=True)
        if isinstance(self.data, object) and len(self.data[0]) == 2:
            self.data, self.lengths = zip(*self.data)
            self.data = np.array(self.data, dtype=float)
            self.lengths = np.array(self.lengths, dtype=float)
        else:
            self.lengths = None

        self.labels = np.load(labels_file_path)
        self.ids = np.load(ids_file_path)
        self.static_data = np.load(static_file_path)
        self.ages = [age for age, _ in self.static_data]
        # self.min_age = min(self.ages)
        # self.max_age = max(self.ages)
        self.dtype = torch.float32
        self.transform = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        id = self.ids[idx]
        label = torch.tensor([label], dtype=self.dtype)
        sample = torch.tensor(np.array(sample), dtype=self.dtype)
        static_data = self.static_data[idx]
        static_data = torch.tensor(static_data, dtype=self.dtype)
        if self.transform:
            sample = self.transform(sample)
        id = torch.tensor([id], dtype=self.dtype)
        return_vals = [sample, label, id, static_data]
        if self.lengths:
            length = self.lengths[idx]
            length = torch.tensor(length, dtype=self.dtype)
            return_vals.append(length)

        return tuple(return_vals)


def compile_grud_dataset(
    data, labels, lengths, ids, dems, length_required=2, max_length=None
):
    """
    Remove instances where ther aren't enough observations.
    """

    data, labels, lengths, dems = remove_short_sequences_numpy(
        data, labels, lengths, dems, length_required
    )
    X = trim_long_sequences(data, lengths, max_length)
    dataset = GRUDDataset(X, labels, ids, lengths, dems)

    return dataset
