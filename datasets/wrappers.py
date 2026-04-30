import torch


class NormalizeGRUDDatasetWrapper:
    def __init__(self, subset, dynamic_transform, age_transform):
        self.subset = subset
        self.dynamic_transform = dynamic_transform
        self.age_transform = age_transform
        self.lengths = subset.lengths
        self.labels = subset.labels

    def __getitem__(self, index):
        sample, label, id, length, static_data = self.subset[index]

        if self.dynamic_transform:
            sample[0] = self.dynamic_transform(sample[0])
            sample[1] = self.dynamic_transform(sample[1])
        if self.age_transform:
            age, gender = static_data
            age = self.age_transform(age)
            static_data = torch.cat((age.unsqueeze(0), gender.unsqueeze(0)))
        return sample, label, id, length, static_data

    def __len__(self):
        return len(self.subset)


class WrapperDataset:
    def __init__(self, subset, dynamic_transform, age_transform):
        self.subset = subset
        self.dynamic_transform = dynamic_transform
        self.age_transform = age_transform

    def __getitem__(self, index):
        sample, label, idx, static_data = self.subset[index]

        if self.dynamic_transform:
            sample = self.dynamic_transform(sample)
        if self.age_transform:
            age, gender = static_data
            age = self.age_transform(age)
            static_data = torch.cat((age.unsqueeze(0), gender.unsqueeze(0)))
        return sample, label, idx, static_data

    def __len__(self):
        return len(self.subset)
