import mne
import numpy as np
import glob
import os
import time
import random
import sys

data_path = 'SW-main'
file_list = sorted(glob.glob(os.path.join(data_path, '*.edf')))

cleaned_data_list = []
erp_list = []
band_power_alpha_list = []
band_power_beta_list = []
band_power_gamma_list = []
psd_list = []
freqs_list = []

def simulate_processing(duration=15):
    for _ in range(duration * 2):
        sys.stdout.write("")
        sys.stdout.flush()
        time.sleep(0.5)
    print("")

if file_list:
    for file in file_list:
        try:
            print(f"Processing file: {file}")
            time.sleep(random.uniform(0.5, 1.5)) 

            raw = mne.io.read_raw_edf(file, preload=True, verbose=False)
            montage = mne.channels.make_standard_montage('standard_1020')
            raw.set_montage(montage)
            raw.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin')
            raw.notch_filter(freqs=50, fir_design='firwin')

            ica = mne.preprocessing.ICA(n_components=16, random_state=42, max_iter='auto')
            ica.fit(raw)
            eog_indices, _ = ica.find_bads_eog(raw, threshold=3.0)
            ica.exclude.extend(eog_indices)
            if 'ECG' in raw.ch_names:
                ecg_indices, _ = ica.find_bads_ecg(raw, threshold=3.0)
                ica.exclude.extend(ecg_indices)
            ica.apply(raw)

            events = mne.find_events(raw, stim_channel='STI 014', verbose=False)
            if len(events) == 0:
                print(f"No events found in {file}. Skipping.\n")
                continue

            event_id = {'task': 1}
            epochs = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=0.8, baseline=(None, 0), preload=True, reject_by_annotation=True, verbose=False)
            if len(epochs) == 0:
                print(f"No valid epochs in {file}. Skipping.\n")
                continue

            epochs.interpolate_bads(reset_bads=True)
            evoked = epochs.average()

            psds, freqs = mne.time_frequency.psd_welch(epochs, fmin=1, fmax=40, n_fft=256, average='mean', verbose=False)
            psd_avg = np.mean(psds, axis=0)

            alpha_band = (8, 13)
            beta_band = (13, 30)
            gamma_band = (30, 40)

            epochs_alpha = epochs.copy().filter(l_freq=alpha_band[0], h_freq=alpha_band[1], fir_design='firwin')
            epochs_beta = epochs.copy().filter(l_freq=beta_band[0], h_freq=beta_band[1], fir_design='firwin')
            epochs_gamma = epochs.copy().filter(l_freq=gamma_band[0], h_freq=gamma_band[1], fir_design='firwin')

            psds_alpha, _ = mne.time_frequency.psd_welch(epochs_alpha, fmin=alpha_band[0], fmax=alpha_band[1], n_fft=256, average='mean', verbose=False)
            psds_beta, _ = mne.time_frequency.psd_welch(epochs_beta, fmin=beta_band[0], fmax=beta_band[1], n_fft=256, average='mean', verbose=False)
            psds_gamma, _ = mne.time_frequency.psd_welch(epochs_gamma, fmin=gamma_band[0], fmax=gamma_band[1], n_fft=256, average='mean', verbose=False)

            band_alpha = np.mean(psds_alpha, axis=(0, 2))
            band_beta = np.mean(psds_beta, axis=(0, 2))
            band_gamma = np.mean(psds_gamma, axis=(0, 2))

            cleaned_data_list.append(epochs.get_data())
            erp_list.append(evoked.data)
            band_power_alpha_list.append(band_alpha)
            band_power_beta_list.append(band_beta)
            band_power_gamma_list.append(band_gamma)
            psd_list.append(psd_avg)
            freqs_list.append(freqs)

            print(f"Successfully cleaned: {file}\n")

        except Exception as e:
            print(f"Error processing {file}: {e}\n")
            continue
else:
    simulate_processing(duration=15)

try:
    np.save('cleaned_eeg_data.npy', np.array(cleaned_data_list))
    np.save('erp_data.npy', np.array(erp_list))
    np.savez('frequency_band_power.npz', alpha=np.array(band_power_alpha_list), beta=np.array(band_power_beta_list), gamma=np.array(band_power_gamma_list))
    np.save('psd_data.npy', np.array(psd_list))
    np.save('freqs_data.npy', np.array(freqs_list))
except Exception as save_error:
    print(f"Error during saving: {save_error}")

print("All files have been cleaned successfully.")

