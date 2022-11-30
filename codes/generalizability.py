import matplotlib.pyplot as plt
import numpy as np

def generalizability(id: str, L_1_data, L_2_data, L_3_data, L_4_data):
    plt.figure(facecolor='lightgray')
    plt.title("Generalizability")
    plt.xlabel("epochs")
    plt.ylabel("acc")
    L_1 = plt.plot(
        np.array([i+1 for i in range(len(L_1_data["generalization"]))]), np.array([float(data['acc']) for data in L_1_data["generalization"]]),
    )
    L_2 = plt.plot(
        np.array([i+1 for i in range(len(L_2_data["generalization"]))]), np.array([float(data['acc']) for data in L_2_data["generalization"]]),
    )
    L_3 = plt.plot(
        np.array([i+1 for i in range(len(L_3_data["generalization"]))]), np.array([float(data['acc']) for data in L_3_data["generalization"]]),
    )
    L_4 = plt.plot(
        np.array([i+1 for i in range(len(L_4_data["generalization"]))]), np.array([float(data['acc']) for data in L_4_data["generalization"]]),
    )
    plt.legend((L_1[0], L_2[0], L_3[0], L_4[0]), ("L_1", "L_2", "L_3", "L_4"), loc=2)
    import os
    try:
        os.mkdir(f"result_graph/{id}")
    except FileExistsError:
        pass
    plt.savefig(f"result_graph/{id}/generalizability.png")
