import matplotlib.pyplot as plt
import numpy as np

def entropy(id: str, L_1_data, L_2_data, L_3_data, L_4_data):
    plt.figure(facecolor='lightgray')
    plt.title("Sender Entropy")
    plt.xlabel("epochs")
    plt.ylabel("entropy value")
    L_1 = plt.plot(
        np.array([i+1 for i in range(len(L_1_data["test"]))]), np.array([float(data["sender_entropy"]) for data in L_1_data['test']]),
    )
    L_2 = plt.plot(
        np.array([i+1 for i in range(len(L_2_data["test"]))]), np.array([float(data["sender_entropy"]) for data in L_2_data['test']]),
    )
    L_3 = plt.plot(
        np.array([i+1 for i in range(len(L_3_data["test"]))]), np.array([float(data["sender_entropy"]) for data in L_3_data['test']]),
    )
    L_4 = plt.plot(
        np.array([i+1 for i in range(len(L_4_data["test"]))]), np.array([float(data["sender_entropy"]) for data in L_4_data['test']]),
    )
    plt.legend((L_1[0], L_2[0], L_3[0], L_4[0]), ("L_1", "L_2", "L_3", "L_4"), loc=1)
    import os
    try:
        os.mkdir(f"result_graph/{id}")
    except FileExistsError:
        pass
    plt.savefig(f"result_graph/{id}/entropy.png")