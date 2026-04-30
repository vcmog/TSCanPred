import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import numpy as np


def custom_collate_function_grud(batch):
    """Custom collate function to handle data with or without seq lengths."""
    # check if lengths are included in the batch
    inputs, labels, ids, _, static_data = zip(*batch)
    labels = torch.tensor(labels)
    ids = torch.tensor(ids)
    static_data = torch.tensor(np.array(static_data))
    seqs_in_position = [[patient[i] for patient in inputs] for i in range(4)]
    padded_seqs = [
        pad_sequence(seqs, batch_first=True, padding_value=0)
        for seqs in seqs_in_position
    ]
    inputs = torch.stack(padded_seqs, dim=1)
    if not torch.is_tensor(inputs):
        inputs = torch.stack(inputs)

    return (inputs, labels, ids, static_data)


def custom_collate_function(batch):
    """Custom collate function to handle data with or without seq lengths."""
    # check if lengths are included in the batch
    unzipped_batch = list(zip(*batch))
    inputs = unzipped_batch[0]
    labels = unzipped_batch[1]
    ids = unzipped_batch[2]
    static_data = unzipped_batch[3]

    first_time_dim = inputs[0].shape[0]
    all_same = all(
        input.shape[0] == first_time_dim for input in inputs
    )  # if all time dimensions not same, needs padding

    labels = torch.tensor(labels)
    ids = torch.tensor(ids)
    static_data = torch.stack(static_data)
    if not all_same:
        reversed_inputs = [seq.flip(0) for seq in inputs]
        inputs = pad_sequence(
            [seq.clone().detach() for seq in reversed_inputs],
            batch_first=True,
            padding_value=0,
        )
        inputs = inputs.flip(1)
    if not torch.is_tensor(inputs):
        inputs = torch.stack(inputs)
    if len(unzipped_batch) == 5:
        print("Using to lengths to pack seqeunce.")
        lengths = unzipped_batch[4]
        lengths = torch.tensor(lengths)
        packed_input_data = pack_padded_sequence(
            inputs, lengths, batch_first=True, enforce_sorted=False
        )
        return (packed_input_data, labels, lengths)
    return (inputs, labels, ids, static_data)
