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
    
    fig, ax = plt.subplots(nrows=2, sharex='col', gridspec_kw={'height_ratios': (6,1)})
    fig.set_facecolor('lightgray')
    fig.suptitle(f"Unigram entropy (natt={settings['natt']},nval={settings['nval']},cvoc={settings['cvoc']},clen={settings['clen']})")
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

    fig, ax = plt.subplots(nrows=2, sharex='col', gridspec_kw={'height_ratios': (6,1)})
    fig.set_facecolor('lightgray')
    fig.suptitle(f"Bigram entropy (natt={settings['natt']},nval={settings['nval']},cvoc={settings['cvoc']},clen={settings['clen']})")
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
    
    plt.figure(facecolor='lightgray')
    plt.title(f"Topsim (natt={settings['natt']},nval={settings['nval']},cvoc={settings['cvoc']},clen={settings['clen']})")
    plt.ylabel("topsim")
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
    plt.title(f"Change of Acc (natt={settings['natt']},nval={settings['nval']},cvoc={settings['cvoc']},clen={settings['clen']})")
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

def average_ease_of_learning_freezed_receiver(ids, L_raw_datas, settings):
    plt.figure(facecolor='lightgray')
    plt.title(f"Ease of Learning (freezed receiver) (natt={settings['natt']},nval={settings['nval']},cvoc={settings['cvoc']},clen={settings['clen']})")
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
    plt.savefig(f"averaged_result/{ids[0]}~{ids[-1]}/ease_of_learning_freezed_receiver.png", dpi=500)

def average_ease_of_learning_freezed_sender(ids, L_raw_datas, settings):
    plt.figure(facecolor='lightgray')
    plt.title(f"Ease of Learning (freezed sender) (natt={settings['natt']},nval={settings['nval']},cvoc={settings['cvoc']},clen={settings['clen']})")
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
    plt.savefig(f"averaged_result/{ids[0]}~{ids[-1]}/ease_of_learning_freezed_sender.png", dpi=500)

def average_sender_entropy(ids, L_raw_datas, settings):
    plt.figure(facecolor='lightgray')
    plt.title(f"Sender Entropy (natt={settings['natt']},nval={settings['nval']},cvoc={settings['cvoc']},clen={settings['clen']})")
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
    plt.savefig(f"averaged_result/{ids[0]}~{ids[-1]}/entropy.png", dpi=500)

def average_generalizability(ids, L_raw_datas, settings):
    plt.figure(facecolor='lightgray')
    plt.title(f"Generalizability (natt={settings['natt']},nval={settings['nval']},cvoc={settings['cvoc']},clen={settings['clen']})")
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

def average_ngram_counts(ids, settings):
    counts_unigrams = []
    counts_bigrams = []
    for id in ids:
        with open(f"model/{id}/counts_unigram.txt", "rb") as f:
            counts_unigrams.append(pickle.load(f))
        with open(f"model/{id}/counts_bigram.txt", "rb") as f:
            counts_bigrams.append(pickle.load(f))
    
    plt.figure(facecolor='lightgray')
    plt.title(f"Unigram Counts (natt={settings['natt']},nval={settings['nval']},cvoc={settings['cvoc']},clen={settings['clen']})")
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
    plt.legend((plot[0] for plot in plots), ("L_1", "L_2", "L_3", "L_4"), loc=1)
    plt.savefig(f"averaged_result/{ids[0]}~{ids[-1]}/ngram_unigram.png", dpi=500)

    plt.figure(facecolor='lightgray')
    plt.title(f"Bigram Counts (natt={settings['natt']},nval={settings['nval']},cvoc={settings['cvoc']},clen={settings['clen']})")
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
    plt.legend((plot[0] for plot in plots), ("L_1", "L_2", "L_3", "L_4"), loc=1)
    plt.savefig(f"averaged_result/{ids[0]}~{ids[-1]}/ngram_bigram.png", dpi=500)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ids', required=True, nargs="*", type=str)
    args = parser.parse_args()
    settings = dict()

    # failed runs are excluded
    ids = []
    skipped = 0
    for id in args.ids:
        files = glob.glob(f"result_md/{id}*.md")
        if 'natt' not in settings:
            numbers = re.findall(r'\d+', files[0])
            settings['natt'], settings['nval'], settings['cvoc'], settings['clen'] = [numbers[i] for i in [1,2,3,4]]
        f = open(files[0], 'r')
        md_text = f.read()
        if "***** FAILED *****" in md_text:
            print(f"{id} will be skipped.", flush=True)
            skipped += 1
        else:
            ids.append(id)
        f.close()
    print(f"Sum: {skipped} runs will be skipped.")

    import os
    try:
        os.mkdir(f"averaged_result/{ids[0]}~{ids[-1]}")
    except FileExistsError:
        pass

    # Need to change according to values of n-gram entropy
    unigram_ylims = [0, 0.1, 3.2, 3.35]
    bigram_ylims = [0, 0.1, 6.2, 6.8]
    average_ngram_entropy(ids, unigram_ylims, bigram_ylims, settings)
    average_topsim(ids, settings)

    L_raw_datas = extract_L_raw_datas(ids)
    average_change_of_acc(ids, L_raw_datas, settings)
    average_ease_of_learning_freezed_receiver(ids, L_raw_datas, settings)
    average_ease_of_learning_freezed_sender(ids, L_raw_datas, settings)
    average_sender_entropy(ids, L_raw_datas, settings)
    average_generalizability(ids, L_raw_datas, settings)

    average_ngram_counts(ids, settings)
