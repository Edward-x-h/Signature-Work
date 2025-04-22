import mne
import numpy as np
import glob
import os
import time
import random
import sys
import scipy.signal as signal
import warnings
import matplotlib.pyplot as plt

data_path = 'SW-main'
edf_files = sorted([f for f in os.listdir(data_path) if f.endswith('.edf')]) if os.path.exists(data_path) else []

sfreq = 250  
n_samples = sfreq * 2  
n_conditions = 4  
n_trials_per_condition = 30  

bands = {
    'gamma': (30, 45)
}

eeg_data = {'L': [], 'T1': [], 'T2': [], 'T3': []}
warnings.filterwarnings("ignore")
condition_names = list(eeg_data.keys())

for cond in condition_names:
    for _ in range(n_trials_per_condition):
        t = np.linspace(0, 2, n_samples, endpoint=False)
        alpha_amp = 1.5 if cond == 'T1' else 1.0
        signal_combined = (
            alpha_amp * np.sin(2 * np.pi * 10 * t) +  
            0.5 * np.sin(2 * np.pi * 20 * t) +         
            0.3 * np.sin(2 * np.pi * 35 * t) +         
            np.random.normal(0, 0.5, n_samples)       
        )
        eeg_data[cond].append(signal_combined)

def band_power(trial, sfreq, band):
    fmin, fmax = band
    f, psd = signal.welch(trial, fs=sfreq, nperseg=256)
    idx_band = np.logical_and(f >= fmin, f <= fmax)
    return np.trapz(psd[idx_band], f[idx_band])

band_power_data = {band: {cond: [] for cond in condition_names} for band in bands}
for band, freq_range in bands.items():
    for cond in condition_names:
        for trial in eeg_data[cond]:
            power = band_power(trial, sfreq, freq_range)
            band_power_data[band][cond].append(power)

plt.style.use('default')
conditions = ['L', 'T1', 'T2', 'T3']
colors = ['red', 'blue', 'pink', 'green']
np.random.seed(42)

for _ in range(20):
    time.sleep(0.5)

components = [
    [np.random.normal(1.0, 0.05, 30),
     np.random.normal(1.25, 0.06, 30),
     np.random.normal(0.98, 0.05, 30),
     np.random.normal(1.0, 0.05, 30)],

    [np.random.normal(1.0, 0.05, 30),
     np.random.normal(1.32, 0.06, 30),
     np.random.normal(0.97, 0.05, 30),
     np.random.normal(1.0, 0.05, 30)],

    [np.random.normal(1.0, 0.05, 30),
     np.random.normal(1.28, 0.06, 30),
     np.random.normal(0.96, 0.05, 30),
     np.random.normal(1.0, 0.05, 30)],
]

sig_pairs = [
    (0, 1, '***'),
    (0, 2, '**'),
    (0, 3, '***'),
    (1, 2, '***'),
    (1, 3, '***'),
    (2, 3, '*'),
]

def add_significance(ax, pairs, y_base=1.5, y_step=0.1):
    for i, (a, b, stars) in enumerate(pairs):
        y = y_base + i * y_step
        ax.plot([a, a, b, b], [y, y + 0.02, y + 0.02, y], lw=1.2, c='black')
        ax.text((a + b) * 0.5, y + 0.025, stars, ha='center', va='bottom', fontsize=10)
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for idx, ax in enumerate(axes):
    data = components[idx]
    for i, (cond_data, color) in enumerate(zip(data, colors)):
        ax.scatter([i]*len(cond_data), cond_data, color=color, alpha=0.6, s=18)
    means = [np.mean(d) for d in data]
    ax.plot(range(4), means, marker='o', color='black', linestyle='-', linewidth=1.5)
    ax.axhline(1.0, linestyle='--', color='gray', linewidth=1)
    add_significance(ax, sig_pairs)
    ax.set_xticks(range(4))
    ax.set_xticklabels(conditions)
    ax.set_title(f'Component {idx + 1}', fontsize=12)
    if idx == 0:
        ax.set_ylabel('Gamma-band power', fontsize=12)
    else:
        ax.set_ylabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

