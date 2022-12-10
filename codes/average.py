import argparse
import matplotlib.pyplot as plt
import numpy as np
import glob
import re
import pickle
from organize_data import extract_one_model_data

def average_ngram_entropy(ids):
    ngram_entropy_data = [
        # L_1
        { "unigram": [], "bigram": [] },
        # L_2
        { "unigram": [], "bigram": [] },
        # L_3
        { "unigram": [], "bigram": [] },
        # L_4
        { "unigram": [], "bigram": [] },
    ]
    for id in ids:
        files = glob.glob(f"result_md/{id}*.md")
        f = open(files[0], 'r')
        md_text = f.read()
        f.close()
        match = re.search(r'### N-gram entropy\n\n\|\| unigram \| bigram \|\n\|-----\|-----\|-----\|\n\| \$L_1\$ \| (.*?) \| (.*?) \|\n\| \$L_2\$ \| (.*?) \| (.*?) \|\n\|\n\| \$L_3\$ \| (.*?) \| (.*?) \|\n\| \$L_4\$ \| (.*?) \| (.*?) \|\n', md_text)
        ngram_entropy_data[0]["unigram"].append(float(match.group(1)))
        ngram_entropy_data[0]["bigram"].append(float(match.group(2)))
        ngram_entropy_data[1]["unigram"].append(float(match.group(3)))
        ngram_entropy_data[1]["bigram"].append(float(match.group(4)))
        ngram_entropy_data[2]["unigram"].append(float(match.group(5)))
        ngram_entropy_data[2]["bigram"].append(float(match.group(6)))
        ngram_entropy_data[3]["unigram"].append(float(match.group(7)))
        ngram_entropy_data[3]["bigram"].append(float(match.group(8)))
    
    plt.figure(facecolor='lightgray')
    plt.title("Unigram entropy")
    plt.ylabel("entropy")
    plt.bar(
        ["L_1", "L_2", "L_3", "L_4"],
        [np.average(ngram_entropy_data[i]["unigram"]) for i in range(4)],
        yerr=[np.std(ngram_entropy_data[i]["unigram"], ddof=1) / np.sqrt(len(ids)) for i in range(4)],
        capsize=10,
    )
    plt.savefig(f"averaged_result/{ids[0]}~{ids[-1]}/unigram_entropy.png")

    plt.figure(facecolor='lightgray')
    plt.title("Bigram entropy")
    plt.ylabel("entropy")
    plt.bar(
        ["L_1", "L_2", "L_3", "L_4"],
        [np.average(ngram_entropy_data[i]["bigram"]) for i in range(4)],
        yerr=[np.std(ngram_entropy_data[i]["bigram"], ddof=1) / np.sqrt(len(ids)) for i in range(4)],
        capsize=10,
    )
    plt.savefig(f"averaged_result/{ids[0]}~{ids[-1]}/bigram_entropy.png")

def average_topsim(ids):
    topsim_data = [
        # L_1
        { "topsim": [] },
        # L_2
        { "topsim": [] },
        # L_3
        { "topsim": [] },
        # L_4
        { "topsim": [] },
    ]
    for id in ids:
        files = glob.glob(f"result_md/{id}*.md")
        f = open(files[0], 'r')
        md_text = f.read()
        f.close()
        match = re.search(r'### Topsim\n\n\|\| Spearman Correlation \|\n\|-----\|-----\|\n\| \$L_1\$ \| (.*?) \|\n\| \$L_2\$ \| (.*?) \|\n\|\n\| \$L_3\$ \| (.*?) \|\n\| \$L_4\$ \| (.*?) \|\n', md_text)
        topsim_data[0]["topsim"].append(float(match.group(1)))
        topsim_data[1]["topsim"].append(float(match.group(2)))
        topsim_data[2]["topsim"].append(float(match.group(3)))
        topsim_data[3]["topsim"].append(float(match.group(4)))
    
    plt.figure(facecolor='lightgray')
    plt.title("Topsim")
    plt.ylabel("topsim")
    plt.bar(
        ["L_1", "L_2", "L_3", "L_4"],
        [np.average(topsim_data[i]["topsim"]) for i in range(4)],
        yerr=[np.std(topsim_data[i]["topsim"], ddof=1) / np.sqrt(len(ids)) for i in range(4)],
        capsize=10,
    )
    plt.savefig(f"averaged_result/{ids[0]}~{ids[-1]}/topsim.png")

# L_raw_datas: the list of numbers of `L_raw_data` generated in each experiment.
def extract_L_raw_datas(ids):
    L_raw_datas = []
    for id in ids:
        L_raw_data = [
            {
                "test": [],
                "generalization": [],
            } for _ in range(12)
        ]
        files = glob.glob(f"result/{id}*.txt")
        f = open(files[0], 'r')
        raw_text = f.read()
        f.close()

        for i in range(12):
            extract_one_model_data(
                re.compile(r"--------------------L_{0} training start--------------------((.|\s)*?)--------------------L_{0} training end--------------------".format(i+1)),
                raw_text,
                L_raw_data[i],
            )
        L_raw_datas.append(L_raw_data)
    return L_raw_datas

# Return the index when count reaches 1 for the first time
def get_limit_index(count):
    for i, v in enumerate(count):
        if v <= 1:
            return i
    return len(count)

def average_change_of_acc(ids, L_raw_datas):
    plt.figure(facecolor='lightgray')
    plt.title("Change of Acc")
    plt.xlabel("epochs")
    plt.ylabel("acc")
    max_epochs = [max([len(L_raw_datas[i][j]["test"]) for i in range(len(L_raw_datas))]) for j in range(4)] # max epochs for each L_1 ~ L_4
    plots = []
    for j in range(4):
        acc = [0 for _ in range(max_epochs[j])]
        std = [[] for _ in range(max_epochs[j])]
        count = [0 for _ in range(max_epochs[j])]
        for i in range(len(L_raw_datas)):
            for k, data in enumerate(L_raw_datas[i][j]["test"]):
                acc[k] += float(data["acc"])
                count[k] += 1
                std[k].append(float(data["acc"]))
        limit_index = get_limit_index(count)
        acc = np.array([a / c for a, c in zip(acc, count)])[:limit_index]
        std = np.array([(np.std(array, ddof=1) / np.sqrt(c)) if c >= 2 else 0.0 for array, c in zip(std, count)])[:limit_index]
        plots.append(plt.plot(np.array([i+1 for i in range(max_epochs[j])])[:limit_index], acc))
        plt.fill_between(np.array([i+1 for i in range(max_epochs[j])])[:limit_index], acc+std, acc-std, alpha=0.25)
    plt.legend((plot[0] for plot in plots), ("L_1", "L_2", "L_3", "L_4"), loc=2)
    plt.savefig(f"averaged_result/{ids[0]}~{ids[-1]}/change_of_acc.png")

def average_ease_of_learning_freezed_receiver(ids, L_raw_datas):
    plt.figure(facecolor='lightgray')
    plt.title("Ease of Learning (freezed receiver)")
    plt.xlabel("epochs")
    plt.ylabel("acc")
    max_epochs = [max([len(L_raw_datas[i][j]["test"]) for i in range(len(L_raw_datas))]) for j in [4, 6, 8, 10]]
    plots = []
    for j, max_epoch in zip([4, 6, 8, 10], max_epochs):
        acc = [0 for _ in range(max_epoch)]
        std = [[] for _ in range(max_epoch)]
        count = [0 for _ in range(max_epoch)]
        for i in range(len(L_raw_datas)):
            for k, data in enumerate(L_raw_datas[i][j]["test"]):
                acc[k] += float(data["acc"])
                count[k] += 1
                std[k].append(float(data["acc"]))
        limit_index = get_limit_index(count)
        acc = np.array([a / c for a, c in zip(acc, count)])[:limit_index]
        std = np.array([(np.std(array, ddof=1) / np.sqrt(c)) if c >= 2 else 0.0 for array, c in zip(std, count)])[:limit_index]
        plots.append(plt.plot(np.array([i+1 for i in range(max_epoch)])[:limit_index], acc))
        plt.fill_between(np.array([i+1 for i in range(max_epoch)])[:limit_index], acc+std, acc-std, alpha=0.25)
    plt.legend((plot[0] for plot in plots), ("L_5", "L_7", "L_9", "L_11"), loc=2)
    plt.savefig(f"averaged_result/{ids[0]}~{ids[-1]}/ease_of_learning_freezed_receiver.png")

def average_ease_of_learning_freezed_sender(ids, L_raw_datas):
    plt.figure(facecolor='lightgray')
    plt.title("Ease of Learning (freezed sender)")
    plt.xlabel("epochs")
    plt.ylabel("acc")
    max_epochs = [max([len(L_raw_datas[i][j]["test"]) for i in range(len(L_raw_datas))]) for j in [5, 7, 9, 11]]
    plots = []
    for j, max_epoch in zip([5, 7, 9, 11], max_epochs):
        acc = [0 for _ in range(max_epoch)]
        std = [[] for _ in range(max_epoch)]
        count = [0 for _ in range(max_epoch)]
        for i in range(len(L_raw_datas)):
            for k, data in enumerate(L_raw_datas[i][j]["test"]):
                acc[k] += float(data["acc"])
                count[k] += 1
                std[k].append(float(data["acc"]))
        limit_index = get_limit_index(count)
        acc = np.array([a / c for a, c in zip(acc, count)])[:limit_index]
        std = np.array([(np.std(array, ddof=1) / np.sqrt(c)) if c >= 2 else 0.0 for array, c in zip(std, count)])[:limit_index]
        plots.append(plt.plot(np.array([i+1 for i in range(max_epoch)])[:limit_index], acc))
        plt.fill_between(np.array([i+1 for i in range(max_epoch)])[:limit_index], acc+std, acc-std, alpha=0.25)
    plt.legend((plot[0] for plot in plots), ("L_6", "L_8", "L_10", "L_12"), loc=2)
    plt.savefig(f"averaged_result/{ids[0]}~{ids[-1]}/ease_of_learning_freezed_sender.png")

def average_sender_entropy(ids, L_raw_datas):
    plt.figure(facecolor='lightgray')
    plt.title("Sender Entropy")
    plt.xlabel("epochs")
    plt.ylabel("entropy value")
    max_epochs = [max([len(L_raw_datas[i][j]["test"]) for i in range(len(L_raw_datas))]) for j in range(4)]
    plots = []
    for j in range(4):
        sender_entropy = [0 for _ in range(max_epochs[j])]
        std = [[] for _ in range(max_epochs[j])]
        count = [0 for _ in range(max_epochs[j])]
        for i in range(len(L_raw_datas)):
            for k, data in enumerate(L_raw_datas[i][j]["test"]):
                sender_entropy[k] += float(data["sender_entropy"])
                count[k] += 1
                std[k].append(float(data["sender_entropy"]))
        limit_index = get_limit_index(count)
        sender_entropy = np.array([a / c for a, c in zip(sender_entropy, count)])[:limit_index]
        std = np.array([(np.std(array, ddof=1) / np.sqrt(c)) if c >= 2 else 0.0 for array, c in zip(std, count)])[:limit_index]
        plots.append(plt.plot(np.array([i+1 for i in range(max_epochs[j])])[:limit_index], sender_entropy))
        plt.fill_between(np.array([i+1 for i in range(max_epochs[j])])[:limit_index], sender_entropy+std, sender_entropy-std, alpha=0.25)
    plt.legend((plot[0] for plot in plots), ("L_1", "L_2", "L_3", "L_4"), loc=1)
    plt.savefig(f"averaged_result/{ids[0]}~{ids[-1]}/entropy.png")

def average_generalizability(ids, L_raw_datas):
    plt.figure(facecolor='lightgray')
    plt.title("Generalizability")
    plt.xlabel("epochs")
    plt.ylabel("acc")
    max_epochs = [max([len(L_raw_datas[i][j]["generalization"]) for i in range(len(L_raw_datas))]) for j in range(4)]
    plots = []
    for j in range(4):
        acc = [0 for _ in range(max_epochs[j])]
        std = [[] for _ in range(max_epochs[j])]
        count = [0 for _ in range(max_epochs[j])]
        for i in range(len(L_raw_datas)):
            for k, data in enumerate(L_raw_datas[i][j]["generalization"]):
                acc[k] += float(data["acc"])
                count[k] += 1
                std[k].append(float(data["acc"]))
        limit_index = get_limit_index(count)
        acc = np.array([a / c for a, c in zip(acc, count)])[:limit_index]
        with np.errstate(all='ignore'):
            std = np.array([(np.std(array, ddof=1) / np.sqrt(c)) if c >= 2 else 0.0 for array, c in zip(std, count)])[:limit_index]
        plots.append(plt.plot(np.array([i+1 for i in range(max_epochs[j])])[:limit_index], acc))
        plt.fill_between(np.array([i+1 for i in range(max_epochs[j])])[:limit_index], acc+std, acc-std, alpha=0.25)
    plt.legend((plot[0] for plot in plots), ("L_1", "L_2", "L_3", "L_4"), loc=2)
    plt.savefig(f"averaged_result/{ids[0]}~{ids[-1]}/generalizability.png")

def average_ngram_counts(ids):
    counts_unigrams = []
    counts_bigrams = []
    for id in ids:
        with open(f"model/{id}/counts_unigram.txt", "rb") as f:
            counts_unigrams.append(pickle.load(f))
        with open(f"model/{id}/counts_bigram.txt", "rb") as f:
            counts_bigrams.append(pickle.load(f))
    
    plt.figure(facecolor='lightgray')
    plt.title("Unigram Counts")
    plt.xlabel("log(Rank)")
    plt.ylabel("log(Counts)")
    vocab_size = len(counts_unigrams[0][0])
    averaged_sorted_counts_unigrams = [np.array([0.0 for _ in range(vocab_size)]) for _ in range(4)]
    for i in range(4):
        with np.errstate(divide="ignore"):
            for j in range(len(ids)):
                averaged_sorted_counts_unigrams[i] += np.log(np.array(sorted(counts_unigrams[j][i], reverse=True))) / len(ids)
    plots = []
    for i in range(4):
        plots.append(plt.plot(
            np.array([np.log(j+1) for j in range(len(averaged_sorted_counts_unigrams[i]))]),
            averaged_sorted_counts_unigrams[i],
        ))
    plt.legend((plot[0] for plot in plots), ("L_1", "L_2", "L_3", "L_4"), loc=1)
    plt.savefig(f"averaged_result/{ids[0]}~{ids[-1]}/ngram_unigram.png")

    plt.figure(facecolor='lightgray')
    plt.title("Bigram Counts")
    plt.xlabel("log(Rank)")
    plt.ylabel("log(Counts)")
    averaged_sorted_counts_bigrams = [np.array([0.0 for _ in range(vocab_size * vocab_size)]) for _ in range(4)]
    for i in range(4):
        with np.errstate(divide="ignore"):
            for j in range(len(ids)):
                averaged_sorted_counts_bigrams[i] += np.log(np.array(sorted(counts_bigrams[j][i], reverse=True))) / len(ids)
    plots = []
    for i in range(4):
        plots.append(plt.plot(
            np.array([np.log(j+1) for j in range(len(averaged_sorted_counts_bigrams[i]))]),
            averaged_sorted_counts_bigrams[i],
        ))
    plt.legend((plot[0] for plot in plots), ("L_1", "L_2", "L_3", "L_4"), loc=1)
    plt.savefig(f"averaged_result/{ids[0]}~{ids[-1]}/ngram_bigram.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ids', required=True, nargs="*", type=str)
    args = parser.parse_args()

    import os
    try:
        os.mkdir(f"averaged_result/{args.ids[0]}~{args.ids[-1]}")
    except FileExistsError:
        pass

    average_ngram_entropy(args.ids)
    average_topsim(args.ids)

    L_raw_datas = extract_L_raw_datas(args.ids)
    average_change_of_acc(args.ids, L_raw_datas)
    average_ease_of_learning_freezed_receiver(args.ids, L_raw_datas)
    average_ease_of_learning_freezed_sender(args.ids, L_raw_datas)
    average_sender_entropy(args.ids, L_raw_datas)
    average_generalizability(args.ids, L_raw_datas)

    average_ngram_counts(args.ids)
