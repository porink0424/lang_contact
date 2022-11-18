# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import torch
from torch.utils.data import DataLoader
import egg.core as core
from data import enumerate_attribute_value, split_train_test, one_hotify, ScaledDataset
from archs import Sender, Receiver, PlusOneWrapper
from loss import DiffLoss

# TODO: Eliminate unused params
def get_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_attributes', type=int, default=4, help='')
    parser.add_argument('--n_values', type=int, default=4, help='')
    parser.add_argument('--data_scaler', type=int, default=100)
    parser.add_argument('--stats_freq', type=int, default=0)
    parser.add_argument('--baseline', type=str, choices=['no', 'mean', 'builtin'], default='mean')
    parser.add_argument('--density_data', type=int,
                        default=0, help='no sampling if equal 0')

    parser.add_argument('--sender_hidden', type=int, default=50,
                        help='Size of the hidden layer of Sender (default: 10)')
    parser.add_argument('--receiver_hidden', type=int, default=50,
                        help='Size of the hidden layer of Receiver (default: 10)')

    parser.add_argument('--sender_entropy_coeff', type=float, default=1e-2,
                        help="Entropy regularisation coeff for Sender (default: 1e-2)")

    parser.add_argument('--sender_cell', type=str, default='rnn')
    parser.add_argument('--receiver_cell', type=str, default='rnn')
    parser.add_argument('--sender_emb', type=int, default=10,
                        help='Size of the embeddings of Sender (default: 10)')
    parser.add_argument('--receiver_emb', type=int, default=10,
                        help='Size of the embeddings of Receiver (default: 10)')
    parser.add_argument('--early_stopping_thr', type=float, default=0.99999,
                        help="Early stopping threshold on accuracy (defautl: 0.99999)")

    args = core.init(arg_parser=parser, params=params)
    return args

def main(params):
    import copy
    opts = get_params(params)
    device = opts.device

    # print settings
    print(opts)

    # make data, and divide into train data and test data
    full_data = enumerate_attribute_value(opts.n_attributes, opts.n_values)
    train, test = split_train_test(full_data, 0.1)
    
    # one-hotify data
    train, test = \
        one_hotify(train, opts.n_attributes, opts.n_values), \
        one_hotify(test, opts.n_attributes, opts.n_values)

    # make data into Dataset, DataLoader
    # TODO: Should we use ScaledDataset????
    train, test = ScaledDataset(train, opts.data_scaler), ScaledDataset(test)
    train_loader, test_loader = \
        DataLoader(train, batch_size=opts.batch_size), \
        DataLoader(test, batch_size=len(test))
    
    n_dim = opts.n_attributes * opts.n_values

    # make a Sender
    if opts.sender_cell in ['lstm', 'rnn', 'gru']:
        # linear layer changes n_inputs to n_hidden
        sender = Sender(n_inputs=n_dim, n_hidden=opts.sender_hidden)
        # Reinforce Wrapper for Sender
        # return sequence, logits, entropy
        sender = core.RnnSenderReinforce(
            agent=sender,
            vocab_size=opts.vocab_size,
            embed_dim=opts.sender_emb,
            hidden_size=opts.sender_hidden,
            max_len=opts.max_len,
            force_eos=False,
            cell=opts.sender_cell
        )
        # In EGG, value 0 means EOS. To avoid this, sender is wrapped by PlusOneWrapper
        # TODO: Should we use PlusOneWrapper????
        sender = PlusOneWrapper(sender)
    else:
        raise ValueError(f'Unknown sender cell, {opts.sender_cell}')

    # make a Receiver
    if opts.receiver_cell in ['lstm', 'rnn', 'gru']:
        receiver = Receiver(n_hidden=opts.receiver_hidden, n_outputs=n_dim)
        # Reinforce Wrapper for a deterministic Receiver
        receiver = core.RnnReceiverDeterministic(
            receiver,
            opts.vocab_size + 1,
            opts.receiver_emb,
            opts.receiver_hidden,
            cell=opts.receiver_cell
        )
    else:
        raise ValueError(f'Unknown receiver cell, {opts.receiver_cell}')

    # calculate a cross_entropy
    loss = DiffLoss(opts.n_attributes, opts.n_values)

    # Implement Sender/Receiver game via Reinforce
    baseline = {
        'no': core.baselines.NoBaseline, 
        'mean': core.baselines.MeanBaseline, 
        'builtin': core.baselines.BuiltInBaseline
    }[opts.baseline]
    game = core.SenderReceiverRnnReinforce(
        sender,
        receiver,
        loss,
        sender_entropy_coeff=opts.sender_entropy_coeff,
        receiver_entropy_coeff=0.0,
        length_cost=0.0,
        baseline_type=baseline,
    )
    optimizer = torch.optim.Adam(game.parameters(), lr=opts.lr)

    # training
    early_stopper = core.EarlyStopperAccuracy(opts.early_stopping_thr, validation=True) # check on test_data, and may stop train.
    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=test_loader,
        callbacks=[
            core.ConsoleLogger(as_json=True, print_train_loss=False),
            early_stopper,
        ]
    )
    trainer.train(n_epochs=opts.n_epochs)

    # try to generate messages
    # TODO: sequence does not seem to change? is it ok?
    batch = torch.tensor([test[0][0].tolist() for _ in range(20)] + [test[1][0].tolist() for _ in range(20)])
    sequence, logits, entropy = sender(batch)
    print(sequence)
    # print(logits)
    # print(entropy)

    print('--------------------End--------------------')

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])