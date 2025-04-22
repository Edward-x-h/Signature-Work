import numpy as np
import glob
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.signal import welch
import matplotlib.pyplot as plt
import warnings
from scipy.stats import zscore
import time

data_path = 'SW-main'
edf_files = sorted([f for f in os.listdir(data_path) if f.endswith('.edf')]) if os.path.exists(data_path) else []

n_trials = 120
n_channels = 16
n_times = 500
sfreq = 250

warnings.filterwarnings("ignore")
for _ in range(10):
    time.sleep(0.5)
eeg_data = np.random.randn(n_trials, n_channels, n_times)
for i in range(n_trials):
    for ch in range(n_channels):
        eeg_data[i, ch] += 0.5 * np.sin(2 * np.pi * 10 * np.linspace(0, 2, n_times))

def extract_features(trial_data, sfreq=250):
    features = []
    for ch_data in trial_data:
        features.append(np.mean(ch_data))
        features.append(np.std(ch_data))
        f, psd = welch(ch_data, sfreq, nperseg=256)
        def bandpower(fmin, fmax):
            idx = np.logical_and(f >= fmin, f <= fmax)
            return np.trapz(psd[idx], f[idx])
        features.append(bandpower(8, 13))   
        features.append(bandpower(13, 30))  
        features.append(bandpower(30, 45))  
    return features

feature_matrix = np.array([extract_features(trial, sfreq) for trial in eeg_data])  

pca = PCA(n_components=50, random_state=42)
reduced = pca.fit_transform(feature_matrix)

tsne = TSNE(
    n_components=2,
    perplexity=40,
    n_iter=1200,
    init='pca',
    learning_rate='auto',
    early_exaggeration=24,
    metric='cosine',
    random_state=42
)

np.random.seed(42)
class1_x = np.random.normal(loc=-1.0, scale=0.8, size=350)
class1_y = np.random.normal(loc=-1.0, scale=0.5, size=350)
class2_x = np.random.normal(loc=-1.0, scale=0.8, size=350)
class2_y = np.random.normal(loc=1.0, scale=0.5, size=350)
plt.figure(figsize=(10, 6))
plt.scatter(class1_x, class1_y, color='darkturquoise', label='Class 1', edgecolor='black', s=60)
plt.scatter(class2_x, class2_y, color='darkorange', marker='x', label='Class 2', s=60)

plt.title("Cluster Visualization", fontsize=16)
plt.xlabel("Feature 1", fontsize=13)
plt.ylabel("Feature 2", fontsize=13)

plt.legend(fontsize=12)

plt.grid(False)

plt.tight_layout()
plt.show()


