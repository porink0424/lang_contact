import argparse
import torch
from torch.utils.data import DataLoader
import egg.core as core
from data import enumerate_attribute_value, split_train_test, one_hotify, ScaledDataset
from archs import Sender, Receiver, PlusOneWrapper, Freezer
from utils import Evaluator
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
    senders, receivers, freezed_senders, freezed_receivers = [], [], [], []

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
    train, validation, test = ScaledDataset(train, opts.data_scaler), ScaledDataset(copy.deepcopy(train)), ScaledDataset(test)
    train_loader, validation_loader, test_loader = \
        DataLoader(train, batch_size=opts.batch_size), \
        DataLoader(validation, batch_size=len(validation)), \
        DataLoader(test, batch_size=len(test))
    
    n_dim = opts.n_attributes * opts.n_values

    for lang_idx in [1, 2]:
        # make a sender
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
            sender = PlusOneWrapper(sender)
        else:
            raise ValueError(f'Unknown sender cell, {opts.sender_cell}')

        # make a Receiver
        if opts.receiver_cell in ['lstm', 'rnn', 'gru']:
            # linear layer changes n_hidden to n_outputs
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
        
        # Implement Sender/Receiver game via Reinforce
        loss = DiffLoss(opts.n_attributes, opts.n_values)
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
        trainer = core.Trainer(
            game=game,
            optimizer=optimizer,
            train_data=train_loader,
            validation_data=validation_loader,
            callbacks=[
                core.ConsoleLogger(as_json=True, print_train_loss=False),
                core.EarlyStopperAccuracy(opts.early_stopping_thr, validation=True),
                Evaluator(
                    [("generalization", test_loader, DiffLoss(opts.n_attributes, opts.n_values))],
                    device,
                    freq=1
                )
            ]
        )
        print(f'--------------------L_{lang_idx} training start--------------------')
        trainer.train(n_epochs=opts.n_epochs)
        print(f'--------------------L_{lang_idx} training end--------------------')
    
        # save the model
        print(f"L_{lang_idx} saving...")
        torch.save(sender, f"model/{opts.id}/L_{lang_idx}-sender.pth")
        torch.save(receiver, f"model/{opts.id}/L_{lang_idx}-receiver.pth")
        print("Done!")
        senders.append(sender)
        receivers.append(receiver)
        freezed_senders.append(Freezer(copy.deepcopy(sender)))
        freezed_receivers.append(Freezer(copy.deepcopy(receiver)))

    for lang_idx in [3, 4]:
        # contact languages setup
        contact_loss = DiffLoss(opts.n_attributes, opts.n_values)
        contact_baseline = {
            'no': core.baselines.NoBaseline, 
            'mean': core.baselines.MeanBaseline, 
            'builtin': core.baselines.BuiltInBaseline
        }[opts.baseline]
        sender = copy.deepcopy(senders[lang_idx - 3])
        receiver = copy.deepcopy(receivers[1 if lang_idx == 3 else 0])
        contact_game = core.SenderReceiverRnnReinforce(
            sender,
            receiver,
            contact_loss,
            sender_entropy_coeff=opts.sender_entropy_coeff,
            receiver_entropy_coeff=0.0,
            length_cost=0.0,
            baseline_type=contact_baseline,
        )
        contact_optimizer = torch.optim.Adam(contact_game.parameters(), lr=opts.lr)
    
        # contact languages training
        contact_trainer = core.Trainer(
            game=contact_game,
            optimizer=contact_optimizer,
            train_data=train_loader,
            validation_data=validation_loader,
            callbacks=[
                core.ConsoleLogger(as_json=True, print_train_loss=False),
                core.EarlyStopperAccuracy(opts.early_stopping_thr, validation=True),
                Evaluator(
                    [("generalization", test_loader, DiffLoss(opts.n_attributes, opts.n_values))],
                    device,
                    freq=1
                )
            ]
        )
        print(f'--------------------L_{lang_idx} training start--------------------')
        contact_trainer.train(n_epochs=opts.n_epochs)
        print(f'--------------------L_{lang_idx} training end--------------------')

        # save the model
        print(f"L_{lang_idx} saving...")
        torch.save(sender, f"model/{opts.id}/L_{lang_idx}-sender.pth")
        torch.save(receiver, f"model/{opts.id}/L_{lang_idx}-receiver.pth")
        print("Done!")
        freezed_senders.append(Freezer(copy.deepcopy(sender)))
        freezed_receivers.append(Freezer(copy.deepcopy(receiver)))
    
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
            'sender': copy.deepcopy(sender),
            'receiver': freezed_receivers[0],
            'train_data': train_loader,
        },
        {
            # P_{1, newR}
            'sender': freezed_senders[0],
            'receiver': copy.deepcopy(receiver),
            # In freezed sender mode, to prevent receivers from learning too quickly, we use non-scaled data
            'train_data': validation_loader,
        },
        {
            # P_{2, newS}
            'sender': copy.deepcopy(sender),
            'receiver': freezed_receivers[1],
            'train_data': train_loader,
        },
        {
            # P_{2, newR}
            'sender': freezed_senders[1],
            'receiver': copy.deepcopy(receiver),
            'train_data': validation_loader,
        },
        {
            # P_{3, newS}
            'sender': copy.deepcopy(sender),
            'receiver': freezed_receivers[2],
            'train_data': train_loader,
        },
        {
            # P_{3, newR}
            'sender': freezed_senders[2],
            'receiver': copy.deepcopy(receiver),
            'train_data': validation_loader,
        },
        {
            # P_{4, newS}
            'sender': copy.deepcopy(sender),
            'receiver': freezed_receivers[3],
            'train_data': train_loader,
        },
        {
            # P_{4, newR}
            'sender': freezed_senders[3],
            'receiver': copy.deepcopy(receiver),
            'train_data': validation_loader,
        },
    ]
    for lang_idx, new_pair in enumerate(new_pairs):
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
        trainer = core.Trainer(
            game=game,
            optimizer=optimizer,
            train_data=new_pair['train_data'],
            validation_data=validation_loader,
            callbacks=[
                core.ConsoleLogger(as_json=True, print_train_loss=False),
                core.EarlyStopperAccuracy(opts.early_stopping_thr, validation=True),
            ]
        )

        print(f'--------------------L_{lang_idx+5} training start--------------------')
        trainer.train(n_epochs=opts.n_epochs // 2)
        print(f'--------------------L_{lang_idx+5} training end--------------------')

    print('--------------------End--------------------')

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])