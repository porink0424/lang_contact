# This source is coded with great reference to
# facebookresearch/EGG/egg/zoo/compo_vs_generalization/
# (https://github.com/facebookresearch/EGG/tree/main/egg/zoo/compo_vs_generalization),
# which are licensed under the MIT license
# (https://github.com/facebookresearch/EGG/blob/main/LICENSE).

import matplotlib.pyplot as plt
import numpy as np

def entropy(id: str, L_raw_data):
    plt.figure(facecolor='lightgray')
    plt.title("Sender Entropy")
    plt.xlabel("epochs")
    plt.ylabel("entropy value")
    plots = [
        plt.plot(
            np.array([i+1 for i in range(len(L_raw_data[j]["test"]))]), np.array([float(raw_data["sender_entropy"]) for raw_data in L_raw_data[j]['test']]),
        ) for j in range(4)
    ]
    plt.legend((plot[0] for plot in plots), ("L_1", "L_2", "L_3", "L_4"), loc=1)
    import os
    try:
        os.mkdir(f"result_graph/{id}")
    except FileExistsError:
        pass
    plt.savefig(f"result_graph/{id}/entropy.png")
