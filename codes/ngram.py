import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
from typing import List, Dict

def ngram(no_cuda: bool, id: str, vocab_size: int) -> Dict[str, List[float]]:
    res = {
        "unigram_entropy": [],
        "bigram_entropy": [],
    }
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
    
    ###################
    #     unigram     #
    ###################  

    # calculate unigram
    counts_unigram = [[0 for _ in range(vocab_size)] for _ in range(4)]
    for i in range(4):
        sequence = sequences[i]
        for sentence in sequence:
            for word in sentence:
                counts_unigram[i][word.item() - 1] += 1
    
    f = open(f"model/{id}/counts_unigram.txt", "wb")
    pickle.dump(counts_unigram, f)
    f.close()

    # visualize unigram
    plt.figure(facecolor='lightgray')
    plt.title("Unigram Counts")
    plt.xlabel("log(Rank)")
    plt.ylabel("log(Counts)")
    plots = []
    for i in range(4):
        sorted_count_unigram = sorted(counts_unigram[i], reverse=True)
        with np.errstate(divide="ignore"):
            plots.append(plt.plot(
                np.array([np.log(i+1) for i in range(len(sorted_count_unigram))]),
                np.log(np.array(sorted_count_unigram)),
            ))
    plt.legend((l[0] for l in plots), ("L_1", "L_2", "L_3", "L_4"), loc=1)
    plt.savefig(f"result_graph/{id}/ngram_unigram.png")

    # calculate unigram entropy
    for count in counts_unigram:
        res["unigram_entropy"].append(entropy(count, base=2))

    ##################
    #     bigram     #
    ##################

    # calculate bigram
    # (w_1, w_2) <-> w_1 * vocab_size + w_2
    counts_bigram = [[0 for _ in range(vocab_size * vocab_size)] for _ in range(4)]
    for i in range(4):
        sequence = sequences[i]
        for sentence in sequence:
            for j in range(len(sentence) - 1):
                counts_bigram[i][(sentence[j].item() - 1) * vocab_size + (sentence[j+1].item() - 1)] += 1
    
    f = open(f"model/{id}/counts_bigram.txt", "wb")
    pickle.dump(counts_bigram, f)
    f.close()

    # visualize bigram
    plt.figure(facecolor='lightgray')
    plt.title("Bigram Counts")
    plt.xlabel("log(Rank)")
    plt.ylabel("log(Counts)")
    plots = []
    for i in range(4):
        sorted_count_bigram = sorted(counts_bigram[i], reverse=True)
        with np.errstate(divide="ignore"):
            plots.append(plt.plot(
                np.array([np.log(i+1) for i in range(len(sorted_count_bigram))]),
                np.log(np.array(sorted_count_bigram)),
            ))
    plt.legend((l[0] for l in plots), ("L_1", "L_2", "L_3", "L_4"), loc=1)
    plt.savefig(f"result_graph/{id}/ngram_bigram.png")

    # calculate unigram entropy
    for count in counts_bigram:
        res["bigram_entropy"].append(entropy(count, base=2))

    return res
