import numpy as np
import json
import matplotlib.pyplot as plt

with open("mermin_5q.json") as r:
    raw = json.load(r)

x = np.array(raw["x"])
y = np.array(raw["y"])

plt.plot(x / np.pi * 180, y)
plt.axhline(4, color="k", linestyle='dashed', label="Local Realism Bound")
plt.axhline(-4, color="k", linestyle='dashed')
plt.axhline(16, color="red", linestyle='dashed', label="Quantum Bound")
plt.axhline(-16, color="red", linestyle='dashed')

plt.xlabel(r"$\theta$ [degrees]")
plt.ylabel("Result")
plt.grid()
plt.legend()
plt.title(f"Mermin Inequality [5Q]\nMax: {y[np.abs(y).argmax()]}")
plt.tight_layout()
plt.savefig("mermin_5q.png", dpi=300)
plt.show()