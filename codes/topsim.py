import torch
import pickle
from scipy.stats import spearmanr
from typing import List
import argparse
from utils import Timer

def topsim(no_cuda: bool, id: str, n_attributes: int, message_length: int) -> List[float]:
    res = []
    device = torch.device("cuda" if (not no_cuda and torch.cuda.is_available()) else "cpu")

    # load input data
    with open(f"model/{id}/train.txt", "rb") as file_train:
        train_data = pickle.load(file_train)
        for i in range(len(train_data)):
            train_data[i] = train_data[i].tolist()
        train_data = torch.tensor(train_data).to(device)
    
    # load sequences
    f = open(f"model/{id}/sequences.txt", "rb")
    sequences = pickle.load(f)
    f.close()
    
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

if __name__ == "__main__":
    # When running this file directly, Calculate `topsim` separately.
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', required=True, type=int)
    parser.add_argument('--n_attributes', required=True, type=int)
    parser.add_argument('--n_values', required=True, type=int)
    parser.add_argument('--vocab_size', required=True, type=int)
    parser.add_argument('--max_len', required=True, type=int)
    args = parser.parse_args()

    with Timer("topsim in the separate place"):
        # load input data
        with open(f"model/{args.id}/train.txt", "rb") as file_train:
            train_data = pickle.load(file_train)
            for i in range(len(train_data)):
                train_data[i] = train_data[i].tolist()
            train_data = torch.tensor(train_data)
        
        # load sequences
        f = open(f"model/{args.id}/sequences.txt", "rb")
        sequences = pickle.load(f)
        f.close()

        # calculate Distance
        topsim_results = []
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
                    for k in range(args.n_attributes):
                        input1_piece = input1[k*(input_length//args.n_attributes):(k+1)*(input_length//args.n_attributes)]
                        input2_piece = input2[k*(input_length//args.n_attributes):(k+1)*(input_length//args.n_attributes)]
                        if input1_piece != input2_piece:
                            count += 1
                    D_input.append(count)

                    # calculate Edit Distance
                    message1 = sequences[sender_index][i]
                    message2 = sequences[sender_index][j]
                    D_msg.append(
                        [message1[l] != message2[l] for l in range(args.max_len)].count(True)
                    )
            correlation, _pvalue = spearmanr(D_input, D_msg)
            topsim_results.append(correlation)
    
    print(topsim_results, flush=True)

    md_file_name = f"result_md/{args.id}--{args.n_attributes}-{args.n_values}-{args.vocab_size}-{args.max_len}.md"
    with open(md_file_name, "r") as md_file:
        md_text = md_file.read()
        for i in range(4):
            md_text = md_text.replace(f"L_{i+1}_topsim_fill_me", str(topsim_results[i]))
    with open(md_file_name, "w") as md_file:
        md_file.write(md_text)
