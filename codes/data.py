import itertools
import torch

# produce all possible input attribute-values
def enumerate_attribute_value(n_attributes, n_values):
    iters = [range(n_values) for _ in range(n_attributes)]
    return list(itertools.product(*iters))

# split all data into train and test at the rate of p_test
def split_train_test(dataset, p_test=0.1, random_seed=7):
    import numpy as np
    random_state = np.random.RandomState(seed=random_seed)

    n_all = len(dataset)
    n_test = int(p_test * n_all)
    permutation = random_state.permutation(n_all)

    train = [dataset[i] for i in permutation[n_test:]]
    test = [dataset[i] for i in permutation[:n_test]]

    assert train and test
    assert len(train) + len(test) == len(dataset)

    return train, test

def one_hotify(data, n_attributes, n_values):
    ret = []
    for config in data:
        z = torch.zeros((n_attributes, n_values))
        for i in range(n_attributes):
            z[i, config[i]] = 1
        ret.append(z.view(-1))
    return ret

class ScaledDataset:
    def __init__(self, examples, scaling_factor=1):
        self.examples = examples
        self.examples_len = len(self.examples)
        self.scaling_factor = scaling_factor
    def __len__(self):
        return self.examples_len * self.scaling_factor
    def __getitem__(self, idx):
        idx = idx % self.examples_len
        return self.examples[idx], torch.zeros(1)
