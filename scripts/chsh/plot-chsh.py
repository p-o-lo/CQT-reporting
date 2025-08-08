import matplotlib.pyplot as plt
import numpy as np
import json

with open("bell-parallel-results.json") as r:
    data: dict[str, float] = json.load(r)

with plt.style.context('fivethirtyeight'):
    plt.bar(data.keys(), data.values())
    plt.title("CHSH Inequalities")
    plt.xlabel("Qubit Pairs")
    plt.ylabel("CHSH Value")
    plt.axhline(2, color="k", linestyle='dashed', label="Local Realism Bound")
    plt.axhline(-2, color="k", linestyle='dashed')
    plt.axhline(2 * np.sqrt(2), color="red", linestyle='dashed', label="Quantum Bound")
    plt.axhline(-2 * np.sqrt(2), color="red", linestyle='dashed')
    plt.ylim(-np.sqrt(2) *2, 2* np.sqrt(2))
    for idx, yval in enumerate(data.values()):
        plt.text(idx, yval / 2, np.round(yval, decimals=2), ha='center')
    plt.legend()
    plt.tight_layout()
    plt.savefig("results.pdf")

plt.show()
