import matplotlib.pyplot as plt
import numpy as np

def ease_of_learning(id: str, L_5_to_12_data):
    plt.figure(facecolor='lightgray')
    plt.title("Ease of Learning")
    plt.xlabel("epochs")
    plt.ylabel("acc")
    plots = (
        plt.plot(
            np.array([i+1 for i in range(len(L_5_to_12_data[j]["test"]))]), np.array([float(data["acc"]) for data in L_5_to_12_data[j]["test"]])
        ) for j in range(8)
    )
    plt.legend((plot[0] for plot in plots), (f"L_{i}" for i in range(5, 12+1)), loc=2)
    import os
    try:
        os.mkdir(f"result_graph/{id}")
    except FileExistsError:
        pass
    plt.savefig(f"result_graph/{id}/ease_of_learning.png")