import torch.nn as nn

class Sender(nn.Module):
    def __init__(self, n_inputs, n_hidden):
        super(Sender, self).__init__()
        self.layer = nn.Linear(n_inputs, n_hidden)

    def forward(self, x):
        x = self.layer(x)
        return x

class Receiver(nn.Module):
    def __init__(self, n_outputs, n_hidden):
        super(Receiver, self).__init__()
        self.layer = nn.Linear(n_hidden, n_outputs)

    # TODO: why should we add _ ???
    def forward(self, x, _):
        x = self.layer(x)
        return x

class PlusOneWrapper(nn.Module):
    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped
    
    def forward(self, *input):
        r1, r2, r3 = self.wrapped(*input)
        return r1 + 1, r2, r3

class Freezer(nn.Module):
    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped
        self.wrapped.eval()

        for p in self.wrapped.parameters():
            p.requires_grad = False

    def train(self, mode):
        pass

    def forward(self, *input):
        return self.wrapped(*input)
