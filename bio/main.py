from Bio import SeqIO
from collections import defaultdict

import matplotlib.pyplot as plt


record = SeqIO.read("../52GJ06_G04_B_F.ab1", "abi")

channels = ["DATA9", "DATA10", "DATA11", "DATA12"]
trace = defaultdict(list)
for c in channels:
    trace[c] = record.annotations["abif_raw"][c]

plt.plot(trace["DATA9"], color="blue")
plt.plot(trace["DATA10"], color="red")
plt.plot(trace["DATA11"], color="green")
plt.plot(trace["DATA12"], color="yellow")
plt.show()
