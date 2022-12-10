import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffLoss(nn.Module):
    def __init__(self, n_attributes, n_values):
        super(DiffLoss, self).__init__()
        self.n_attributes = n_attributes
        self.n_values = n_values
    
    def forward(self, sender_input, _message, _receiver_input, receiver_output, _labels):
        batch_size = sender_input.size(0)
        sender_input = sender_input.view(batch_size, self.n_attributes, self.n_values)
        receiver_output = receiver_output.view(batch_size, self.n_attributes, self.n_values)

        acc = (
            torch.sum(
                (receiver_output.argmax(dim=-1) == sender_input.argmax(dim=-1)).detach(),
                dim=1,
            ) == self.n_attributes
        ).float().mean()

        acc_or = (receiver_output.argmax(dim=-1) == sender_input.argmax(dim=-1)).float().mean()

        receiver_output = receiver_output.view(batch_size * self.n_attributes, self.n_values)
        labels = sender_input.argmax(dim=-1).view(batch_size * self.n_attributes)
        loss = F.cross_entropy(receiver_output, labels, reduction="none") \
            .view(batch_size, self.n_attributes).mean(dim=-1)
        
        return loss, {'acc': acc, 'acc_or': acc_or}
