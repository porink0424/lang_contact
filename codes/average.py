import argparse
import matplotlib.pyplot as plt
import numpy as np
import glob
import re

def ngram_averaged_entropy(ids):
    L_data = [
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

        L_data[0]["unigram"].append(float(match.group(1)))
        L_data[0]["bigram"].append(float(match.group(2)))
        L_data[1]["unigram"].append(float(match.group(3)))
        L_data[1]["bigram"].append(float(match.group(4)))
        L_data[2]["unigram"].append(float(match.group(5)))
        L_data[2]["bigram"].append(float(match.group(6)))
        L_data[3]["unigram"].append(float(match.group(7)))
        L_data[3]["bigram"].append(float(match.group(8)))
    
    plt.figure(facecolor='lightgray')
    plt.title("Unigram entropy")
    plt.ylabel("entropy")
    plt.bar(
        ["L_1", "L_2", "L_3", "L_4"],
        [np.average(L_data[i]["unigram"]) for i in range(4)],
        yerr=[np.std(L_data[i]["unigram"]) for i in range(4)],
    )
    plt.savefig(f"averaged_result/{ids[0]}~{ids[-1]}/unigram_entropy.png")

    plt.figure(facecolor='lightgray')
    plt.title("Bigram entropy")
    plt.ylabel("entropy")
    plt.bar(
        ["L_1", "L_2", "L_3", "L_4"],
        [np.average(L_data[i]["bigram"]) for i in range(4)],
        yerr=[np.std(L_data[i]["bigram"]) for i in range(4)],
    )
    plt.savefig(f"averaged_result/{ids[0]}~{ids[-1]}/bigram_entropy.png")

def topsim_averaged_topsim(ids):
    L_data = [
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

        L_data[0]["topsim"].append(float(match.group(1)))
        L_data[1]["topsim"].append(float(match.group(2)))
        L_data[2]["topsim"].append(float(match.group(3)))
        L_data[3]["topsim"].append(float(match.group(4)))
    
    plt.figure(facecolor='lightgray')
    plt.title("Topsim")
    plt.ylabel("topsim")
    plt.bar(
        ["L_1", "L_2", "L_3", "L_4"],
        [np.average(L_data[i]["topsim"]) for i in range(4)],
        yerr=[np.std(L_data[i]["topsim"]) for i in range(4)],
    )
    plt.savefig(f"averaged_result/{ids[0]}~{ids[-1]}/topsim.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ids', required=True, nargs="*", type=str)
    args = parser.parse_args()

    import os
    try:
        os.mkdir(f"averaged_result/{args.ids[0]}~{args.ids[-1]}")
    except FileExistsError:
        pass

    ngram_averaged_entropy(args.ids)
    topsim_averaged_topsim(args.ids)
