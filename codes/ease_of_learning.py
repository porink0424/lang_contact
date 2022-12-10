import matplotlib.pyplot as plt
import numpy as np

def ease_of_learning(id: str, L_raw_data):
    plt.figure(facecolor='lightgray')
    plt.title("Ease of Learning (freezed receiver)")
    plt.xlabel("epochs")
    plt.ylabel("acc")
    plots = (
        plt.plot(
            np.array([i+1 for i in range(len(L_raw_data[j]["test"]))]), np.array([float(raw_data["acc"]) for raw_data in L_raw_data[j]["test"]])
        ) for j in [4, 6, 8, 10]
    )
    plt.legend((plot[0] for plot in plots), ("L_5", "L_7", "L_9", "L_11"), loc=2)
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
            np.array([i+1 for i in range(len(L_raw_data[j]["test"]))]), np.array([float(raw_data["acc"]) for raw_data in L_raw_data[j]["test"]])
        ) for j in [5, 7, 9, 11]
    )
    plt.legend((plot[0] for plot in plots), ("L_6", "L_8", "L_10", "L_12"), loc=2)
    import os
    try:
        os.mkdir(f"result_graph/{id}")
    except FileExistsError:
        pass
    plt.savefig(f"result_graph/{id}/ease_of_learning_freezed_sender.png")