from collections import defaultdict

import numpy as np
import random


class BatchSampler:
    # define bins by quantiles ensuring mostly equal patients per bin
    def __init__(self, lengths, batch_size, num_bins=4):
        super(BatchSampler).__init__()
        self.lengths = np.array(lengths)
        self.batch_size = batch_size
        self.num_bins = num_bins
        # Get original order of indices
        self.indices = np.arange(len(lengths))

        # Sort indices by length (for binning only)
        sorted_indices = self.indices[np.argsort(self.lengths)]
        sorted_lengths = self.lengths[sorted_indices]

        # Compute bin edges using quantiles (ensuring roughly equal-sized bins)
        bin_edges = np.quantile(sorted_lengths, np.linspace(0, 1, num_bins + 1))

        # Assign each index to a bin while maintaining original order
        bin_assignments = np.searchsorted(bin_edges, self.lengths, side="right") - 1
        bin_assignments = np.clip(bin_assignments, 0, num_bins - 1).flatten()
        # Organize indices by bins while preserving original order
        self.bins = [[] for _ in range(num_bins)]
        for idx, bin_idx in zip(self.indices, bin_assignments):
            self.bins[bin_idx].append(idx)

    def __iter__(self):
        for bin_indices in self.bins:
            for i in range(0, len(bin_indices), self.batch_size):
                batch = bin_indices[i : i + self.batch_size]
                yield batch

    def __len__(self):
        # + batch_size - 1 ensures the last batch is counted even if it is smaller than the batch_size
        return sum(
            (len(bin_indices) + self.batch_size - 1) // self.batch_size
            for bin_indices in self.bins
        )


class BatchSamplerBinbyLength(BatchSampler):
    # This defines bin edges based on the actual min and max
    def __init__(self, lengths, batch_size, num_bins=4):
        super().__init__(lengths, batch_size, num_bins)
        self.lengths = lengths
        self.batch_size = batch_size
        self.num_bins = num_bins
        self.indices = list(range(len(lengths)))
        self.bins = self.create_length_bins(lengths, num_bins)

    def create_length_bins(self, lengths, num_bins):
        """Group sequences into length bins."""
        min_len, max_len = min(lengths), max(lengths)
        bin_width = (max_len - min_len) // num_bins + 1
        bins = defaultdict(list)

        for idx, length in enumerate(lengths):
            bin_idx = (length - min_len) // bin_width
            bins[bin_idx].append(idx)

        return bins

    def __iter__(self):
        bin_keys = list(self.bins.keys())
        random.shuffle(bin_keys)

        for bin_key in bin_keys:

            bin_indices = self.bins[bin_key]

            random.shuffle(bin_indices)

            for i in range(0, len(bin_indices), self.batch_size):
                batch = bin_indices[i : i + self.batch_size]
                yield batch

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size
