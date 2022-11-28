# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import torch
from torch.utils.data import DataLoader
import egg.core as core
from data import enumerate_attribute_value, split_train_test, one_hotify, ScaledDataset
from archs import Sender, Receiver, PlusOneWrapper, Freezer
from loss import DiffLoss
import pickle
import copy

def get_params(params):
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default='id_unknown', help='')
    parser.add_argument('--comment', type=str, default='', help='')
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
    opts = get_params(params)
    device = opts.device
    freezed_senders = []
    freezed_receivers = []

    # print settings
    print(opts)

    # make data, and divide into train data and test data
    full_data = enumerate_attribute_value(opts.n_attributes, opts.n_values)
    train, test = split_train_test(full_data, 0.1)
    
    # one-hotify data
    train, test = \
        one_hotify(train, opts.n_attributes, opts.n_values), \
        one_hotify(test, opts.n_attributes, opts.n_values)
    
    # save inputs
    import os
    try:
        os.mkdir(f"model/{opts.id}")
    except FileExistsError:
        pass
    file_train = open(f"model/{opts.id}/train.txt", "wb")
    pickle.dump(train, file_train)
    file_train.close()
    file_test = open(f"model/{opts.id}/test.txt", "wb")
    pickle.dump(test, file_test)
    file_test.close()

    # make data into Dataset, DataLoader
    # To promote training, ScaledDataset is used.
    train, test = ScaledDataset(train, opts.data_scaler), ScaledDataset(test)
    train_loader, test_loader = \
        DataLoader(train, batch_size=opts.batch_size), \
        DataLoader(test, batch_size=len(test))
    
    n_dim = opts.n_attributes * opts.n_values

    # make a Sender
    if opts.sender_cell in ['lstm', 'rnn', 'gru']:
        # linear layer changes n_inputs to n_hidden
        senders = [Sender(n_inputs=n_dim, n_hidden=opts.sender_hidden) for _ in range(2)]
        # Reinforce Wrapper for Sender
        # return sequence, logits, entropy
        senders = [
            core.RnnSenderReinforce(
                agent=sender,
                vocab_size=opts.vocab_size,
                embed_dim=opts.sender_emb,
                hidden_size=opts.sender_hidden,
                max_len=opts.max_len,
                force_eos=False,
                cell=opts.sender_cell
            ) for sender in senders
        ]
        # In EGG, value 0 means EOS. To avoid this, sender is wrapped by PlusOneWrapper
        senders = [PlusOneWrapper(sender) for sender in senders]
    else:
        raise ValueError(f'Unknown sender cell, {opts.sender_cell}')

    # make a Receiver
    if opts.receiver_cell in ['lstm', 'rnn', 'gru']:
        # linear layer changes n_hidden to n_outputs
        receivers = [Receiver(n_hidden=opts.receiver_hidden, n_outputs=n_dim) for _ in range(2)]
        # Reinforce Wrapper for a deterministic Receiver
        receivers = [
            core.RnnReceiverDeterministic(
                receiver,
                opts.vocab_size + 1,
                opts.receiver_emb,
                opts.receiver_hidden,
                cell=opts.receiver_cell
            ) for receiver in receivers
        ]
    else:
        raise ValueError(f'Unknown receiver cell, {opts.receiver_cell}')

    # calculate a cross_entropy
    losses = [DiffLoss(opts.n_attributes, opts.n_values) for _ in range(2)]

    # Implement Sender/Receiver game via Reinforce
    baseline = [
        {
            'no': core.baselines.NoBaseline, 
            'mean': core.baselines.MeanBaseline, 
            'builtin': core.baselines.BuiltInBaseline
        }[opts.baseline] for _ in range(2)
    ]
    games = [
        core.SenderReceiverRnnReinforce(
            senders[i],
            receivers[i],
            losses[i],
            sender_entropy_coeff=opts.sender_entropy_coeff,
            receiver_entropy_coeff=0.0,
            length_cost=0.0,
            baseline_type=baseline[i],
        ) for i in range(2)
    ]
    optimizers = [torch.optim.Adam(game.parameters(), lr=opts.lr) for game in games]

    # training
    early_stopper = core.EarlyStopperAccuracy(opts.early_stopping_thr, validation=False) # early stoppers check the accuracy on train_data, and may stop train according to it.
    trainers = [
        core.Trainer(
            game=games[i],
            optimizer=optimizers[i],
            train_data=train_loader,
            validation_data=test_loader,
            callbacks=[
                core.ConsoleLogger(as_json=True, print_train_loss=True),
                early_stopper,
            ]
        ) for i in range(2)
    ]
    # TODO: training using multithread
    for i, trainer in enumerate(trainers):
        print(f'--------------------L_{i+1} training start--------------------')
        trainer.train(n_epochs=opts.n_epochs)
        print(f'--------------------L_{i+1} training end--------------------')
    
    # save the model
    for i in range(2):
        print(f"L_{i+1} saving...")
        torch.save(senders[i], f"model/{opts.id}/L_{i+1}-sender.pth")
        torch.save(receivers[i], f"model/{opts.id}/L_{i+1}-receiver.pth")
        print("Done!")
        freezed_senders.append(Freezer(copy.deepcopy(senders[i])))
        freezed_receivers.append(Freezer(copy.deepcopy(receivers[i])))

    # contact languages setup
    contact_losses = [DiffLoss(opts.n_attributes, opts.n_values) for _ in range(2)]
    contact_baseline = [
        {
            'no': core.baselines.NoBaseline, 
            'mean': core.baselines.MeanBaseline, 
            'builtin': core.baselines.BuiltInBaseline
        }[opts.baseline] for _ in range(2)
    ]
    contact_games = [
        core.SenderReceiverRnnReinforce(
            senders[i],
            receivers[j],
            contact_losses[i],
            sender_entropy_coeff=opts.sender_entropy_coeff,
            receiver_entropy_coeff=0.0,
            length_cost=0.0,
            baseline_type=contact_baseline[i],
        ) for (i, j) in [(0, 1), (1, 0)]
    ]
    contact_optimizers = [torch.optim.Adam(game.parameters(), lr=opts.lr) for game in contact_games]
    
    # contact languages training
    contact_trainers = [
        core.Trainer(
            game=contact_games[i],
            optimizer=contact_optimizers[i],
            train_data=train_loader,
            validation_data=test_loader,
            callbacks=[
                core.ConsoleLogger(as_json=True, print_train_loss=True),
                early_stopper,
            ]
        ) for i in range(2)
    ]
    # TODO: training using multithread
    for i, trainer in enumerate(contact_trainers):
        print(f'--------------------L_{i+3} training start--------------------')
        trainer.train(n_epochs=opts.n_epochs)
        print(f'--------------------L_{i+3} training end--------------------')

    # save the model
    for i in range(2):
        print(f"L_{i+3} saving...")
        torch.save(senders[i], f"model/{opts.id}/L_{i+3}-sender.pth")
        torch.save(receivers[i], f"model/{opts.id}/L_{i+3}-receiver.pth")
        print("Done!")
        freezed_senders.append(Freezer(copy.deepcopy(senders[i])))
        freezed_receivers.append(Freezer(copy.deepcopy(receivers[i])))

    # Finally, calculate the ease of learning

    # make a new Sender and a new Receiver
    sender = Sender(n_inputs=n_dim, n_hidden=opts.sender_hidden)
    sender = core.RnnSenderReinforce(
        agent=sender,
        vocab_size=opts.vocab_size,
        embed_dim=opts.sender_emb,
        hidden_size=opts.sender_hidden,
        max_len=opts.max_len,
        force_eos=False,
        cell=opts.sender_cell
    )
    sender = PlusOneWrapper(sender)
    receiver = Receiver(n_hidden=opts.receiver_hidden, n_outputs=n_dim)
    receiver = core.RnnReceiverDeterministic(
        receiver,
        opts.vocab_size + 1,
        opts.receiver_emb,
        opts.receiver_hidden,
        cell=opts.receiver_cell
    )

    # training
    new_pairs = [
        {
            # P_{1, newS}
            'protocol_name': 'L_5',
            'sender': copy.deepcopy(sender),
            'receiver': freezed_receivers[0],
        },
        {
            # P_{1, newR}
            'protocol_name': 'L_6',
            'sender': freezed_senders[0],
            'receiver': copy.deepcopy(receiver),
        },
        {
            # P_{2, newS}
            'protocol_name': 'L_7',
            'sender': copy.deepcopy(sender),
            'receiver': freezed_receivers[1],
        },
        {
            # P_{2, newR}
            'protocol_name': 'L_8',
            'sender': freezed_senders[1],
            'receiver': copy.deepcopy(receiver),
        },
        {
            # P_{3, newS}
            'protocol_name': 'L_9',
            'sender': copy.deepcopy(sender),
            'receiver': freezed_receivers[2],
        },
        {
            # P_{3, newR}
            'protocol_name': 'L_10',
            'sender': freezed_senders[2],
            'receiver': copy.deepcopy(receiver),
        },
        {
            # P_{4, newS}
            'protocol_name': 'L_11',
            'sender': copy.deepcopy(sender),
            'receiver': freezed_receivers[3],
        },
        {
            # P_{4, newR}
            'protocol_name': 'L_12',
            'sender': freezed_senders[3],
            'receiver': copy.deepcopy(receiver),
        },
    ]
    for new_pair in new_pairs:
        loss = DiffLoss(opts.n_attributes, opts.n_values)
        baseline = {
            'no': core.baselines.NoBaseline, 
            'mean': core.baselines.MeanBaseline, 
            'builtin': core.baselines.BuiltInBaseline
        }[opts.baseline]
        game = core.SenderReceiverRnnReinforce(
            new_pair['sender'],
            new_pair['receiver'],
            loss,
            sender_entropy_coeff=opts.sender_entropy_coeff,
            receiver_entropy_coeff=0.0,
            length_cost=0.0,
            baseline_type=baseline,
        )
        optimizer = torch.optim.Adam(game.parameters(), lr=opts.lr)
        early_stopper = core.EarlyStopperAccuracy(opts.early_stopping_thr, validation=False)
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
        print(f'--------------------{new_pair["protocol_name"]} training start--------------------')
        trainer.train(n_epochs=opts.n_epochs)
        print(f'--------------------{new_pair["protocol_name"]} training end--------------------')

    print('--------------------End--------------------')

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])