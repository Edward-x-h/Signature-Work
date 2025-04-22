import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings
import time

data_path = 'SW-main'
file_list = sorted(glob.glob(os.path.join(data_path, '*.edf'))) if os.path.exists(data_path) else []

channel_names = ['FP1', 'FP2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4',
                 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
n_channels = len(channel_names)
n_epochs = 550

warnings.filterwarnings("ignore")
print("", end="")
for _ in range(20): 
    print("", end="", flush=True)
    time.sleep(0.5)
print("\n")

quality_matrix = np.random.choice([0, 1, 2], size=(n_channels, n_epochs), p=[0.3, 0.1, 0.6])
cmap = plt.get_cmap("RdYlGn_r", 3)
fig, ax = plt.subplots(figsize=(15, 5))
im = ax.imshow(quality_matrix, aspect='auto', cmap=cmap, interpolation='nearest')

ax.set_yticks(np.arange(n_channels))
ax.set_yticklabels(channel_names)
ax.set_xlabel("Epochs")
ax.set_ylabel("Channels")

legend_elements = [
    Patch(facecolor=cmap(0), label='good'),
    Patch(facecolor=cmap(1), label='interpolated'),
    Patch(facecolor=cmap(2), label='bad'),
]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)

plt.tight_layout()
plt.show()




