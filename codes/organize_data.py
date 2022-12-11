import re
import glob
from tomark import Tomark
from entropy import entropy
from ngram import ngram
from topsim import topsim
from generalizability import generalizability
from ease_of_learning import ease_of_learning
from change_of_acc import change_of_acc
from utils import Timer
import pickle
import torch

# Extract data from `raw_text` between `--training start--` and `--training end--`,
# and return as `L_raw_data`.
def extract_one_model_data(pattern, raw_text, L_raw_data):
    result = re.search(pattern, raw_text)
    if not result:
        raise ValueError(pattern)
    
    for epoch in re.compile(
        r"\{\"mode\": \"(.*?)\", \"epoch\": (.*?), \"loss\": (.*?), \"acc\": (.*?), \"acc_or\": (.*?), \"sender_entropy\": (.*?), \"receiver_entropy\": (.*?), \"original_loss\": (.*?), \"mean_length\": (.*?)\}"
    ).finditer(result.group(1)):
        data = {
            'loss': epoch.group(3),
            'acc': epoch.group(4),
            'acc_or': epoch.group(5),
            'sender_entropy': epoch.group(6),
            'receiver_entropy': epoch.group(7),
            'original_loss': epoch.group(8),
            'mean_length': epoch.group(9),
        }
        if epoch.group(1) == "test":
            L_raw_data["test"].append(data)
        else:
            raise ValueError()
    
    for epoch in re.compile(
        r"\{\"generalization\": \{\"acc\": (.*?), \"acc_or\": (.*?)\}, \"epoch\": (.*?)\}"
    ).finditer(result.group(1)):
        data = {
            'acc': epoch.group(1),
            'acc_or': epoch.group(2),
        }
        L_raw_data["generalization"].append(data)

# Generate sender's outputs and dump to a file
def generate_sequences(id, no_cuda: bool):
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
        sequences.append(sequence.cpu())
    
    # dump messages
    f = open(f"model/{id}/sequences.txt", "wb")
    pickle.dump(sequences, f)
    f.close()

def main(file_path: str):
    with Timer("read file"):
        f = open(file_path, 'r')
        raw_text = f.read()
        f.close()

        file_name = (file_path.split("/")[1]).split(".")[0]

    # extract experiment settings
    with Timer("extract config"):
        pattern = re.compile(r"Namespace\(baseline='(.*?)', batch_size=(.*?), checkpoint_dir=(.*?), checkpoint_freq=(.*?), comment='(.*?)', cuda=(.*?), data_scaler=(.*?), density_data=(.*?), device=(.*?), early_stopping_thr=(.*?), id='(.*?)', load_from_checkpoint=(.*?), lr=(.*?), max_len=(.*?), n_attributes=(.*?), n_epochs=(.*?), n_values=(.*?), no_cuda=(.*?), optimizer='(.*?)', preemptable=(.*?), random_seed=(.*?), receiver_cell='(.*?)', receiver_emb=(.*?), receiver_hidden=(.*?), sender_cell='(.*?)', sender_emb=(.*?), sender_entropy_coeff=(.*?), sender_hidden=(.*?), stats_freq=(.*?), tensorboard=(.*?), tensorboard_dir='(.*?)', validation_freq=(.*?), vocab_size=(.*?)\)")
        result = re.search(pattern, raw_text)
        if not result:
            raise ValueError()
        config = { # all values are string
            'baseline': result.group(1),
            'batch_size': result.group(2),
            'checkpoint_dir': result.group(3),
            'checkpoint_freq': result.group(4),
            'comment': result.group(5),
            'cuda': result.group(6),
            'data_scaler': result.group(7),
            'density_data': result.group(8),
            'device': result.group(9),
            'early_stopping_thr': result.group(10),
            'id': result.group(11),
            'load_from_checkpoint': result.group(12),
            'lr': result.group(13),
            'max_len': result.group(14),
            'n_attributes': result.group(15),
            'n_epochs': result.group(16),
            'n_values': result.group(17),
            'no_cuda': result.group(18),
            'optimizer': result.group(19),
            'preemptable': result.group(20),
            'random_seed': result.group(21),
            'receiver_cell': result.group(22),
            'receiver_emb': result.group(23),
            'receiver_hidden': result.group(24),
            'sender_cell': result.group(25),
            'sender_emb': result.group(26),
            'sender_entropy_coeff': result.group(27),
            'sender_hidden': result.group(28),
            'stats_freq': result.group(29),
            'tensorboard': result.group(30),
            'tensorboard_dir': result.group(31),
            'validation_freq': result.group(32),
            'vocab_size': result.group(33),
        }

    # extract results about L_1 ~ L_12
    with Timer("extract results"):
        L_raw_data = [
            {
                "test": [],
                "generalization": [],
            } for _ in range(12)
        ]
        for i in range(12):
            extract_one_model_data(
                re.compile(r"--------------------L_{0} training start--------------------((.|\s)*?)--------------------L_{0} training end--------------------".format(i+1)),
                raw_text,
                L_raw_data[i],
            )
    
    # change of acc
    with Timer("change of acc"):
        change_of_acc(config['id'], L_raw_data)

    # entropy
    with Timer("entropy"):
        entropy(config['id'], L_raw_data)
    
    # generate messages and dump to files
    with Timer("generate messages"):
        generate_sequences(config['id'], config["no_cuda"] == "True")

    # ngram
    with Timer("ngram entropy"):
        ngram_entropy = ngram(config['id'], int(config['vocab_size']))

    # ↓ topsim will be calculated in `topsim.sh`, so now will be skipped.
    # # topsim
    # with Timer("topsim"):
    #     topsim_result = topsim(config["no_cuda"] == "True", config['id'], int(config['n_attributes']), int(config['max_len']))
    topsim_result = [f"L_{i}_topsim_fill_me" for i in [1,2,3,4]]

    # generalizability
    with Timer("generalizability"):
        generalizability(config['id'], L_raw_data)

    # ease of learning
    with Timer("ease of learning"):
        ease_of_learning(config['id'], L_raw_data)

    # make a markdown file as an organized and visualized result
    with Timer("generate a markdown file"):
        f = open(f"result_md/{file_name}.md", "w")
        transposed_config = []
        for k,v in config.items():
            transposed_config.append({"args": k, "values": v})
        table = Tomark.table(transposed_config)
        md_text = f"# {file_name}\n\n" \
            + (f"### Comment\n\n{config['comment']}\n\n" if config['comment'] else "") \
            + f"### Setting\n\n{table}\n\n"
        
        # Add info related to n-gram entropy
        md_text += f"### N-gram entropy\n\n"
        md_text += "|| unigram | bigram |\n" \
            + "|-----|-----|-----|\n" \
            + f"| $L_1$ | {ngram_entropy['unigram_entropy'][0]} | {ngram_entropy['bigram_entropy'][0]} |\n" \
            + f"| $L_2$ | {ngram_entropy['unigram_entropy'][1]} | {ngram_entropy['bigram_entropy'][1]} |\n" \
            + "|\n" \
            + f"| $L_3$ | {ngram_entropy['unigram_entropy'][2]} | {ngram_entropy['bigram_entropy'][2]} |\n" \
            + f"| $L_4$ | {ngram_entropy['unigram_entropy'][3]} | {ngram_entropy['bigram_entropy'][3]} |\n\n"
        
        # Add info related to topsim
        md_text += f"### Topsim\n\n"
        md_text += "|| Spearman Correlation |\n" \
            + "|-----|-----|\n" \
            + f"| $L_1$ | {topsim_result[0]} |\n" \
            + f"| $L_2$ | {topsim_result[1]} |\n" \
            + "|\n" \
            + f"| $L_3$ | {topsim_result[2]} |\n" \
            + f"| $L_4$ | {topsim_result[3]} |\n\n"

        # Add all graphs
        md_text += f"### Graphs\n\n"
        files = sorted(glob.glob(f"./result_graph/{config['id']}/*"))
        for file in files:
            md_text += f"![{file[1:]}]({file[1:]})\n\n"
        f.write(md_text)
        f.close()

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
