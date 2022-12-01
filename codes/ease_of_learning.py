import matplotlib.pyplot as plt
import numpy as np

def ease_of_learning(id: str, L_5_to_12_data):
    plt.figure(facecolor='lightgray')
    plt.title("Ease of Learning (freezed receiver)")
    plt.xlabel("epochs")
    plt.ylabel("acc")
    plots = (
        plt.plot(
            np.array([i+1 for i in range(len(L_5_to_12_data[j]["test"]))]), np.array([float(data["acc"]) for data in L_5_to_12_data[j]["test"]])
        ) for j in [0, 2, 4, 6]
    )
    plt.legend((plot[0] for plot in plots), (f"L_{i}" for i in [5, 7, 9, 11]), loc=2)
    import os
    try:
        os.mkdir(f"result_graph/{id}")
    except FileExistsError:
        pass
    plt.savefig(f"result_graph/{id}/ease_of_learning_freezed_receiver.png")

    plt.figure(facecolor='lightgray')
    plt.title("Ease of Learning (freezed sender)")
    plt.xlabel("epochs")
    plt.ylabel("acc")
    plots = (
        plt.plot(
            np.array([i+1 for i in range(len(L_5_to_12_data[j]["test"]))]), np.array([float(data["acc"]) for data in L_5_to_12_data[j]["test"]])
        ) for j in [1, 3, 5, 7]
    )
    plt.legend((plot[0] for plot in plots), (f"L_{i}" for i in [6, 8, 10, 12]), loc=2)
    import os
    try:
        os.mkdir(f"result_graph/{id}")
    except FileExistsError:
        pass
    plt.savefig(f"result_graph/{id}/ease_of_learning_freezed_sender.png")