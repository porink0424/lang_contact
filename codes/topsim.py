import torch
import pickle
from scipy.stats import spearmanr
from typing import List

def topsim(no_cuda: bool, id: str, n_attributes: int, message_length: int) -> List[float]:
    res = []
    device = torch.device("cuda" if (not no_cuda and torch.cuda.is_available()) else "cpu")

    # load input data
    with open(f"model/{id}/train.txt", "rb") as file_train:
        train_data = pickle.load(file_train)
        for i in range(len(train_data)):
            train_data[i] = train_data[i].tolist()
        train_data = torch.tensor(train_data).to(device)
    
    # load model
    senders = [
        torch.load(f"model/{id}/L_{lang_idx}-sender.pth").eval() for lang_idx in [1,2,3,4]
    ]

    # generate messages for all inputs
    sequences = []
    for sender in senders:
        sequence, _logits, _entropy = sender(train_data)
        sequences.append(sequence)

    # calculate Distance
    for sender_index in range(4):
        D_input = []
        D_msg = []
        for i in range(len(train_data)):
            for j in range(i+1, len(train_data)):
                input1 = train_data[i].tolist()
                input2 = train_data[j].tolist()

                # calculate Hamming Distance
                count = 0
                input_length = len(input1)
                for k in range(n_attributes):
                    input1_piece = input1[k*(input_length//n_attributes):(k+1)*(input_length//n_attributes)]
                    input2_piece = input2[k*(input_length//n_attributes):(k+1)*(input_length//n_attributes)]
                    if input1_piece != input2_piece:
                        count += 1
                D_input.append(count)

                # calculate Edit Distance
                message1 = sequences[sender_index][i]
                message2 = sequences[sender_index][j]
                D_msg.append(
                    [message1[l] != message2[l] for l in range(message_length)].count(True)
                )
        correlation, _pvalue = spearmanr(D_input, D_msg)
        res.append(correlation)
    
    return res
