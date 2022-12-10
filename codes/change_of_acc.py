import matplotlib.pyplot as plt
import numpy as np

def change_of_acc(id: str, L_raw_data):
    plt.figure(facecolor='lightgray')
    plt.title("Change of Acc")
    plt.xlabel("epochs")
    plt.ylabel("acc")
    plots = [
        plt.plot(
            np.array([i+1 for i in range(len(L_raw_data[j]["test"]))]), np.array([float(raw_data['acc']) for raw_data in L_raw_data[j]["test"]]),
        ) for j in range(4)
    ]
    plt.legend((plot[0] for plot in plots), ("L_1", "L_2", "L_3", "L_4"), loc=2)
    import os
    try:
        os.mkdir(f"result_graph/{id}")
    except FileExistsError:
        pass
    plt.savefig(f"result_graph/{id}/change_of_acc.png")
