import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy

def ngram(no_cuda: bool, id: str, vocab_size: int):
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
    sequences = []
    senders = [
        torch.load(f"model/{id}/L_{i+1}-sender.pth").eval() for i in range(4)
    ]

    # generate messages for all inputs
    for sender in senders:
        sequence, _logits, _entropy = sender(train_data)
        sequences.append(sequence)
    
    ###################
    #     unigram     #
    ###################  

    # calculate unigram
    counts_unigram = [[0 for _ in range(vocab_size)] for _ in range(len(senders))]
    for i in range(len(senders)):
        sequence = sequences[i]
        for sentence in sequence:
            for word in sentence:
                counts_unigram[i][word.item() - 1] += 1

    # visualize unigram
    # graph of the counts for each word
    fig, axs = plt.subplots(2, 2, facecolor='lightgray')
    plt.title("Unigram Counts")
    # extract max count
    max = 0
    for i in range(4):
        for v in counts_unigram[i]:
            if max < v:
                max = v
    for i in range(4):
        ax = axs[1 if i>=2 else 0, i%2]
        ax.set_title(f"L_{i+1}")
        ax.set_xlabel("word")
        ax.set_ylabel("Count for word")
        ax.bar([j for j in range(1, vocab_size+1)], counts_unigram[i])
        ax.set_ylim([0, max+1])
        ax.label_outer()
    plt.savefig(f"result_graph/{id}/unigram_count.png")
    # histogram of the counts
    plt.figure(facecolor='lightgray')
    plt.title("Unigram Counts Histogram")
    plt.xlabel("Counts")
    plt.ylabel("Counts of Counts")
    labels = ["L_1", "L_2", "L_3", "L_4"]
    plt.hist(counts_unigram, stacked=False, label=labels)
    plt.legend()
    plt.savefig(f"result_graph/{id}/unigram_histogram.png")

    # calculate unigram entropy
    for count in counts_unigram:
        res["unigram_entropy"].append(entropy(count, base=2))

    ##################
    #     bigram     #
    ##################

    # calculate bigram
    # 0 ~ vocab_size-1: indices of words,
    # vocab_size: index of BOS
    # vocab_size+1: index of EOS
    # (w_1, w_2) <-> w_1 * (vocab_size+2) + w_2
    counts_bigram = [[0 for _ in range((vocab_size+2) * (vocab_size+2))] for _ in range(len(senders))]
    n_word_pairs = [0 for _ in range(len(senders))]
    for i in range(len(senders)):
        sequence = sequences[i]
        for sentence in sequence:
            for j in range(-1, len(sentence)):
                if j == -1:
                    # w_1 is BOS
                    counts_bigram[i][vocab_size * (vocab_size+2) + (sentence[j+1].item() - 1)] += 1
                elif j == len(sentence) - 1:
                    # w_2 is EOS
                    counts_bigram[i][(sentence[j].item() - 1) * (vocab_size+2) + (vocab_size+1)] += 1
                else:
                    counts_bigram[i][(sentence[j].item() - 1) * (vocab_size+2) + (sentence[j+1].item() - 1)] += 1
                n_word_pairs[i] += 1

    # visualize bigram
    # graph of the counts for each word pair
    fig, axs = plt.subplots(2, 2, facecolor='lightgray')
    plt.title("Bigram Counts")
    # extract max count
    max = 0
    for i in range(4):
        for v in counts_bigram[i]:
            if max < v:
                max = v
    for i in range(4):
        ax = axs[1 if i>=2 else 0, i%2]
        ax.set_title(f"L_{i+1}")
        ax.set_xlabel("word pair")
        ax.set_ylabel("Count for word pair")
        ax.bar([j for j in range(len(counts_bigram[i]))], counts_bigram[i])
        ax.set_ylim([0, max+1])
        ax.label_outer()
    plt.savefig(f"result_graph/{id}/bigram_count.png")
    # histogram of the counts
    plt.figure(facecolor='lightgray')
    plt.title("Bigram Counts Histogram")
    plt.xlabel("Counts")
    plt.ylabel("Counts of Counts")
    labels = ["L_1", "L_2", "L_3", "L_4"]
    plt.hist(counts_bigram, stacked=False, label=labels, bins=30)
    plt.legend()
    plt.savefig(f"result_graph/{id}/bigram_histogram.png")

    # calculate unigram entropy
    for count in counts_bigram:
        res["bigram_entropy"].append(entropy(count, base=2))

    return res