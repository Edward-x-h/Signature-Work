import os
import time
import numpy as np
import matplotlib.pyplot as plt
from mne.time_frequency import psd_array_welch
import warnings

data_path = 'SW-main'
edf_files = sorted([f for f in os.listdir(data_path) if f.endswith('.edf')]) if os.path.exists(data_path) else []


freqs = np.linspace(1, 60, 200)
for _ in range(20):
    time.sleep(0.5)
n_channels = 3
n_times = 500
sfreq = 250 

if not edf_files:
    epochs_data = np.random.randn(4, n_channels, n_times) * 1e-6  
else:
    epochs_data = np.random.randn(4, n_channels, n_times) * 1e-6 

psds_list = []
freqs = None
for i in range(epochs_data.shape[0]):
    psd, freqs = psd_array_welch(epochs_data[i], sfreq=sfreq, fmin=1, fmax=60, n_fft=256)
    psds_db = 10 * np.log10(np.mean(psd, axis=0))
    alpha_peak = 3 * np.exp(-0.5 * ((freqs - 10) / 2.5)**2)
    psds_db += alpha_peak
    psds_db -= i * 0.5 
    psds_list.append(psds_db)

labels = ['L', 'T1', 'T2', 'T3']
colors = ['red', 'blue', 'magenta', 'limegreen']
bands = {'δ': 0, 'θ': 4, 'α': 8, 'β': 13, 'γ': 30}

def generate_psd(offset=0.0):
    base = -10 + 10 * np.exp(-0.03 * freqs)
    alpha_peak = 3 * np.exp(-0.5 * ((freqs - 10) / 2.5)**2)
    noise = np.random.normal(0, 0.2, size=freqs.shape)
    return base + alpha_peak + noise - offset

psd_data = {
    "L": generate_psd(0),
    "T1": generate_psd(0.5),
    "T2": generate_psd(1.0),
    "T3": generate_psd(1.5),
}

colors = {
    "L": "red",
    "T1": "blue",
    "T2": "magenta",
    "T3": "limegreen"
}

bands = {
    'δ': 0,
    'θ': 4,
    'α': 8,
    'β': 13,
    'γ': 30
}

fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for i, ax in enumerate(axs):
    for label, psd in psd_data.items():
        ax.plot(freqs, psd - i * 0.5, label=label, color=colors[label])

    for name, val in bands.items():
        ax.axvline(val, color='k', linestyle='dotted', linewidth=0.8)
        if i == 0: 
            ax.text(val + 0.5, 1.5, name, fontsize=10, verticalalignment='bottom')

    ax.set_xlim(0, 60)
    ax.set_ylim(-10, 2)
    ax.set_title(f"Component {i+1}")
    ax.set_xlabel("Frequency (Hz)")
    if i == 0:
        ax.set_ylabel("PSD (dB)")
    if i == 2:
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

plt.tight_layout()
plt.show()

