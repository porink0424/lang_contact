# This source is coded with great reference to
# facebookresearch/EGG/egg/zoo/compo_vs_generalization/
# (https://github.com/facebookresearch/EGG/tree/main/egg/zoo/compo_vs_generalization),
# which are licensed under the MIT license
# (https://github.com/facebookresearch/EGG/blob/main/LICENSE).

import argparse
import matplotlib.pyplot as plt
import numpy as np
import glob
import re
import pickle
from organize_data import extract_one_model_data

# plt.style.use('tableau-colorblind10')

plt.rcParams["font.size"] = 14
plt.tight_layout()

def average_ngram_entropy(ids, unigram_ylims, bigram_ylims, settings):
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
    
    fig, ax = plt.subplots(nrows=2, sharex='col', gridspec_kw={'height_ratios': (6,1)}, figsize=(4,8))
    fig.subplots_adjust(left=0.16)
    fig.set_facecolor('lightgray')
    fig.suptitle(f"Unigram entropy ({settings['natt']},{settings['nval']},{settings['cvoc']},{settings['clen']})")
    fig.subplots_adjust(hspace=0.05)
    ax[0].set_ylabel("entropy")
    ax[0].set_ylim(unigram_ylims[2], unigram_ylims[3])
    ax[0].bar(
        ["L_1", "L_2", "L_3", "L_4"],
        [np.average(ngram_entropy_data[i]["unigram"]) for i in range(4)],
        yerr=[np.std(ngram_entropy_data[i]["unigram"], ddof=1) / np.sqrt(len(ids)) for i in range(4)],
        capsize=10,
    )
    d = 0.02
    line_args = dict(transform=ax[0].transAxes, lw=1, color='k', clip_on=False, linestyle=':')
    ax[0].plot((-d, 1+d), (0, 0), **line_args)
    ax[0].spines['bottom'].set_visible(False)
    ax[0].tick_params(bottom=False)
    ax[1].set_ylim(unigram_ylims[0], unigram_ylims[1])
    ax[1].bar(
        ["L_1", "L_2", "L_3", "L_4"],
        [np.average(ngram_entropy_data[i]["unigram"]) for i in range(4)],
        yerr=[np.std(ngram_entropy_data[i]["unigram"], ddof=1) / np.sqrt(len(ids)) for i in range(4)],
        capsize=10,
    )
    line_args = dict(transform=ax[1].transAxes, lw=1, color='k', clip_on=False, linestyle=':')
    ax[1].plot((-d, 1+d), (1, 1), **line_args)
    ax[1].spines['top'].set_visible(False)
    ax[1].set_yticks([0])
    ax[1].set_yticklabels([0])
    fig.savefig(f"averaged_result/{ids[0]}~{ids[-1]}/unigram_entropy.png", dpi=500)

    fig, ax = plt.subplots(nrows=2, sharex='col', gridspec_kw={'height_ratios': (6,1)}, figsize=(4,8))
    fig.subplots_adjust(left=0.16)
    fig.set_facecolor('lightgray')
    fig.suptitle(f"Bigram entropy ({settings['natt']},{settings['nval']},{settings['cvoc']},{settings['clen']})")
    fig.subplots_adjust(hspace=0.05)
    ax[0].set_ylabel("entropy")
    ax[0].set_ylim(bigram_ylims[2], bigram_ylims[3])
    ax[0].bar(
        ["L_1", "L_2", "L_3", "L_4"],
        [np.average(ngram_entropy_data[i]["bigram"]) for i in range(4)],
        yerr=[np.std(ngram_entropy_data[i]["bigram"], ddof=1) / np.sqrt(len(ids)) for i in range(4)],
        capsize=10,
    )
    d = 0.02
    line_args = dict(transform=ax[0].transAxes, lw=1, color='k', clip_on=False, linestyle=':')
    ax[0].plot((-d, 1+d), (0, 0), **line_args)
    ax[0].spines['bottom'].set_visible(False)
    ax[0].tick_params(bottom=False)
    ax[1].set_ylim(bigram_ylims[0], bigram_ylims[1])
    ax[1].bar(
        ["L_1", "L_2", "L_3", "L_4"],
        [np.average(ngram_entropy_data[i]["bigram"]) for i in range(4)],
        yerr=[np.std(ngram_entropy_data[i]["bigram"], ddof=1) / np.sqrt(len(ids)) for i in range(4)],
        capsize=10,
    )
    line_args = dict(transform=ax[1].transAxes, lw=1, color='k', clip_on=False, linestyle=':')
    ax[1].plot((-d, 1+d), (1, 1), **line_args)
    ax[1].spines['top'].set_visible(False)
    ax[1].set_yticks([0])
    ax[1].set_yticklabels([0])
    fig.savefig(f"averaged_result/{ids[0]}~{ids[-1]}/bigram_entropy.png", dpi=500)

def average_sender_entropy(ids, settings):
    sender_entropy_data = [
        # L_1
        { "sender_entropy": [] },
        # L_2
        { "sender_entropy": [] },
        # L_3
        { "sender_entropy": [] },
        # L_4
        { "sender_entropy": [] },
    ]
    for id in ids:
        files = glob.glob(f"result/{id}*txt")
        f = open(files[0], 'r')
        raw_text = f.read()
        f.close()
        match = re.search(r'\{\"mode\": \"test\", \"epoch\": .*?, \"loss\": .*?, \"acc\": .*?, \"acc_or\": .*?, \"sender_entropy\": (.*?), .*?\}\n\{\"generalization\": \{.*?\}, .*?\}\n--------------------L_1 training end--------------------', raw_text)
        sender_entropy_data[0]["sender_entropy"].append(float(match.group(1)))
        match = re.search(r'\{\"mode\": \"test\", \"epoch\": .*?, \"loss\": .*?, \"acc\": .*?, \"acc_or\": .*?, \"sender_entropy\": (.*?), .*?\}\n\{\"generalization\": \{.*?\}, .*?\}\n--------------------L_2 training end--------------------', raw_text)
        sender_entropy_data[1]["sender_entropy"].append(float(match.group(1)))
        match = re.search(r'\{\"mode\": \"test\", \"epoch\": .*?, \"loss\": .*?, \"acc\": .*?, \"acc_or\": .*?, \"sender_entropy\": (.*?), .*?\}\n\{\"generalization\": \{.*?\}, .*?\}\n--------------------L_3 training end--------------------', raw_text)
        sender_entropy_data[2]["sender_entropy"].append(float(match.group(1)))
        match = re.search(r'\{\"mode\": \"test\", \"epoch\": .*?, \"loss\": .*?, \"acc\": .*?, \"acc_or\": .*?, \"sender_entropy\": (.*?), .*?\}\n\{\"generalization\": \{.*?\}, .*?\}\n--------------------L_4 training end--------------------', raw_text)
        sender_entropy_data[3]["sender_entropy"].append(float(match.group(1)))
    
    fig = plt.figure(facecolor='lightgray', figsize=(4,8))
    fig.subplots_adjust(left=0.16)
    plt.ylim(0, 0.7)
    plt.title(f"Sender Entropy ({settings['natt']},{settings['nval']},{settings['cvoc']},{settings['clen']})")
    plt.ylabel("entropy")
    plt.bar(
        ["L_1", "L_2", "L_3", "L_4"],
        [np.average(sender_entropy_data[i]["sender_entropy"]) for i in range(4)],
        yerr=[np.std(sender_entropy_data[i]["sender_entropy"], ddof=1) / np.sqrt(len(ids)) for i in range(4)],
        capsize=10,
    )
    plt.savefig(f"averaged_result/{ids[0]}~{ids[-1]}/sender_entropy.png", dpi=500)

def average_topsim(ids, settings):
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
    
    fig = plt.figure(facecolor='lightgray', figsize=(4,8))
    fig.subplots_adjust(left=0.19)
    plt.title(f"Topsim ({settings['natt']},{settings['nval']},{settings['cvoc']},{settings['clen']})")
    plt.ylabel("topsim")
    plt.ylim(0, 0.3)
    plt.bar(
        ["L_1", "L_2", "L_3", "L_4"],
        [np.average(topsim_data[i]["topsim"]) for i in range(4)],
        yerr=[np.std(topsim_data[i]["topsim"], ddof=1) / np.sqrt(len(ids)) for i in range(4)],
        capsize=10,
    )
    plt.savefig(f"averaged_result/{ids[0]}~{ids[-1]}/topsim.png", dpi=500)

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

def average_change_of_acc(ids, L_raw_datas, settings):
    plt.figure(facecolor='lightgray')
    plt.title(f"Change of Acc ({settings['natt']},{settings['nval']},{settings['cvoc']},{settings['clen']})")
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
    plt.legend((plot[0] for plot in plots), ("L_1", "L_2", "L_3", "L_4"), loc=4)
    plt.savefig(f"averaged_result/{ids[0]}~{ids[-1]}/change_of_acc.png", dpi=500)

    plt.figure(facecolor='lightgray')
    plt.title(f"Change of Acc(log scale) ({settings['natt']},{settings['nval']},{settings['cvoc']},{settings['clen']})")
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
        plots.append(plt.plot(np.array([np.log(i+1) for i in range(max_epochs[j])])[:limit_index], acc))
        plt.fill_between(np.array([np.log(i+1) for i in range(max_epochs[j])])[:limit_index], acc+std, acc-std, alpha=0.25)
    plt.legend((plot[0] for plot in plots), ("L_1", "L_2", "L_3", "L_4"), loc=4)
    plt.savefig(f"averaged_result/{ids[0]}~{ids[-1]}/change_of_acc_log.png", dpi=500)

def average_ease_of_learning_frozen_receiver(ids, L_raw_datas, settings):
    plt.figure(facecolor='lightgray')
    plt.title(f"Ease of Learning (frozen receiver) ({settings['natt']},{settings['nval']},{settings['cvoc']},{settings['clen']})")
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
    plt.legend((plot[0] for plot in plots), ("L_5", "L_7", "L_9", "L_11"), loc=4)
    plt.savefig(f"averaged_result/{ids[0]}~{ids[-1]}/ease_of_learning_frozen_receiver.png", dpi=500)

def average_ease_of_learning_frozen_sender(ids, L_raw_datas, settings):
    plt.figure(facecolor='lightgray')
    plt.title(f"Ease of Learning (frozen sender) ({settings['natt']},{settings['nval']},{settings['cvoc']},{settings['clen']})")
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
    plt.legend((plot[0] for plot in plots), ("L_6", "L_8", "L_10", "L_12"), loc=4)
    plt.savefig(f"averaged_result/{ids[0]}~{ids[-1]}/ease_of_learning_frozen_sender.png", dpi=500)

def average_generalizability(ids, L_raw_datas, settings):
    plt.figure(facecolor='lightgray')
    plt.title(f"Generalizability ({settings['natt']},{settings['nval']},{settings['cvoc']},{settings['clen']})")
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
    plt.legend((plot[0] for plot in plots), ("L_1", "L_2", "L_3", "L_4"), loc=4)
    plt.savefig(f"averaged_result/{ids[0]}~{ids[-1]}/generalizability.png", dpi=500)

    plt.figure(facecolor='lightgray')
    plt.title(f"Generalizability(log scale) ({settings['natt']},{settings['nval']},{settings['cvoc']},{settings['clen']})")
    plt.xlabel("log(epochs)")
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
        plots.append(plt.plot(np.array([np.log(i+1) for i in range(max_epochs[j])])[:limit_index], acc))
        plt.fill_between(np.array([np.log(i+1) for i in range(max_epochs[j])])[:limit_index], acc+std, acc-std, alpha=0.25)
    plt.legend((plot[0] for plot in plots), ("L_1", "L_2", "L_3", "L_4"), loc=4)
    plt.savefig(f"averaged_result/{ids[0]}~{ids[-1]}/generalizability_log.png", dpi=500)

def average_ngram_counts(ids, settings):
    counts_unigrams = []
    counts_bigrams = []
    for id in ids:
        with open(f"model/{id}/counts_unigram.txt", "rb") as f:
            counts_unigrams.append(pickle.load(f))
        with open(f"model/{id}/counts_bigram.txt", "rb") as f:
            counts_bigrams.append(pickle.load(f))
    
    plt.figure(facecolor='lightgray')
    plt.title(f"Unigram Counts ({settings['natt']},{settings['nval']},{settings['cvoc']},{settings['clen']})")
    plt.xlabel("log(Rank)")
    plt.ylabel("log(Counts)")
    plots = []
    vocab_size = len(counts_unigrams[0][0])
    for i in range(4):
        ngram_counts = [0.0 for _ in range(vocab_size)]
        std = [[] for _ in range(vocab_size)]
        for j in range(len(ids)):
            with np.errstate(divide="ignore"):
                sorted_count_unigram = np.log(np.array(sorted(counts_unigrams[j][i], reverse=True)))
            for k in range(vocab_size):
                ngram_counts[k] += sorted_count_unigram[k]
                std[k].append(sorted_count_unigram[k])
        ngram_counts = np.array(ngram_counts) / len(ids)
        with np.errstate(invalid="ignore"):
            std = np.array([np.std(array, ddof=1) for array in std])
        plots.append(plt.plot(
            np.array([np.log(k+1) for k in range(vocab_size)]),
            ngram_counts,
        ))
        with np.errstate(invalid="ignore"):
            plt.fill_between(
                np.array([np.log(k+1) for k in range(vocab_size)]),
                ngram_counts + std,
                ngram_counts - std,
                alpha=0.25
            )
    plt.legend((plot[0] for plot in plots), ("L_1", "L_2", "L_3", "L_4"), loc=3)
    plt.savefig(f"averaged_result/{ids[0]}~{ids[-1]}/ngram_unigram.png", dpi=500)

    plt.figure(facecolor='lightgray')
    plt.title(f"Bigram Counts ({settings['natt']},{settings['nval']},{settings['cvoc']},{settings['clen']})")
    plt.xlabel("log(Rank)")
    plt.ylabel("log(Counts)")
    plots = []
    for i in range(4):
        ngram_counts = [0.0 for _ in range(vocab_size * vocab_size)]
        std = [[] for _ in range(vocab_size * vocab_size)]
        for j in range(len(ids)):
            with np.errstate(divide="ignore"):
                sorted_count_bigram = np.log(np.array(sorted(counts_bigrams[j][i], reverse=True)))
            for k in range(vocab_size * vocab_size):
                ngram_counts[k] += sorted_count_bigram[k]
                std[k].append(sorted_count_bigram[k])
        ngram_counts = np.array(ngram_counts) / len(ids)
        with np.errstate(invalid="ignore"):
            std = np.array([np.std(array, ddof=1) for array in std])
        plots.append(plt.plot(
            np.array([np.log(k+1) for k in range(vocab_size * vocab_size)]),
            ngram_counts,
        ))
        with np.errstate(invalid="ignore"):
            plt.fill_between(
                np.array([np.log(k+1) for k in range(vocab_size * vocab_size)]),
                ngram_counts + std,
                ngram_counts - std,
                alpha=0.25
            )
    plt.legend((plot[0] for plot in plots), ("L_1", "L_2", "L_3", "L_4"), loc=3)
    plt.savefig(f"averaged_result/{ids[0]}~{ids[-1]}/ngram_bigram.png", dpi=500)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ids', required=True, nargs="*", type=str)
    args = parser.parse_args()
    settings = dict()

    # failed runs are excluded
    ids = []
    L_1_L_2_failed = 0
    L_3_L_4_failed = 0
    for id in args.ids:
        files = glob.glob(f"result_md/{id}*.md")
        if 'natt' not in settings:
            numbers = re.findall(r'\d+', files[0])
            settings['natt'], settings['nval'], settings['cvoc'], settings['clen'] = [numbers[i] for i in [1,2,3,4]]
        f = open(files[0], 'r')
        md_text = f.read()
        if "***** FAILED *****" in md_text:
            print(f"{id} will be skipped.", flush=True)
            if "L_1 failed: True" in md_text or "L_2 failed: True" in md_text:
                L_1_L_2_failed += 1
            elif "L_3 failed: True" in md_text or "L_4 failed: True" in md_text:
                L_3_L_4_failed += 1
            else:
                raise ValueError()
        else:
            ids.append(id)
        f.close()
    print(f"L_1 or L_2 failed: {L_1_L_2_failed}, L_3 or L_4 failed: {L_3_L_4_failed}", flush=True)
    print(f"Sum: {len(ids)} out of {len(args.ids)} runs successful.", flush=True)

    import os
    try:
        os.mkdir(f"averaged_result/{ids[0]}~{ids[-1]}")
    except FileExistsError:
        pass

    unigram_ylims = [0, 0.1, 2.2, 3.4]
    bigram_ylims = [0, 0.1, 4.4, 6.7]

    average_ngram_entropy(ids, unigram_ylims, bigram_ylims, settings)
    average_sender_entropy(ids, settings)
    average_topsim(ids, settings)

    L_raw_datas = extract_L_raw_datas(ids)
    average_change_of_acc(ids, L_raw_datas, settings)
    average_ease_of_learning_frozen_receiver(ids, L_raw_datas, settings)
    average_ease_of_learning_frozen_sender(ids, L_raw_datas, settings)
    average_generalizability(ids, L_raw_datas, settings)

    average_ngram_counts(ids, settings)
