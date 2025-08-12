import json
import numpy as np
import matplotlib.pyplot as plt

qubits_list = [0,3,6]
n_shots = 1000
datapath = f"data/QFT_sim/QFT_on_{qubits_list}_{n_shots}shots.json"
data = json.load(open(datapath, 'rb'))

dataplot = data['plotparameters']['frequencies']

n_qubits = len(qubits_list)

with plt.style.context('fivethirtyeight'):
    plt.bar(dataplot.keys(), dataplot.values())
    plt.title("QFT")
    plt.xlabel("States")
    plt.xticks(rotation=90)
    plt.ylabel("Counts")
    plt.axhline(n_shots/2**n_qubits, color="k", linestyle='dashed', label="Target Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"QFT_on_{qubits_list}_{n_shots}shots.pdf")

plt.show()