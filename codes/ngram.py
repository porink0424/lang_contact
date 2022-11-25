import torch
import pickle
import matplotlib.pyplot as plt

def unigram(no_cuda: bool, id: str, vocab_size: int):
    device = torch.device("cuda" if (not no_cuda and torch.cuda.is_available()) else "cpu")

    # load input data
    # TODO: もともと使っていたinputだけを使うべきか？
    with open(f"model/{id}/train.txt", "rb") as file_train:
        train_data = pickle.load(file_train)
        for i in range(len(train_data)):
            train_data[i] = train_data[i].tolist()
        train_data = torch.tensor(train_data).to(device)
    with open(f"model/{id}/test.txt", "rb") as file_test:
        test_data = pickle.load(file_test)
        for i in range(len(test_data)):
            test_data[i] = test_data[i].tolist()
        test_data = torch.tensor(test_data).to(device)

    # load model
    sequences = []
    senders = [
        torch.load(f"model/{id}/L_{i+1}-sender.pth").eval() for i in range(4)
    ]

    # generate messages for all inputs
    for sender in senders:
        sequence, _logits, _entropy = sender(train_data)
        sequences.append(sequence)
    
    # calculate unigram
    counts_unigram = [[0 for _ in range(vocab_size)] for _ in range(len(senders))]
    n_words = [0 for _ in range(len(senders))]
    for i in range(len(senders)):
        sequence = sequences[i]
        for sentence in sequence:
            for word in sentence:
                counts_unigram[i][word.item() - 1] += 1
                n_words[i] += 1

    # visualize unigram
    # graph of the counts for each word
    for i in range(len(senders)):
        plt.figure(facecolor='lightgray')
        plt.title("Unigram Counts")
        plt.xlabel("word")
        plt.ylabel("Count for word")
        plt.bar([i for i in range(1, vocab_size+1)], counts_unigram[i])
        plt.savefig(f"result_graph/{id}/unigram_count.png")
    # histogram of the counts
    # TODO: 実装