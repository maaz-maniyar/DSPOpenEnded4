import os
import numpy as np
import scipy.io as sio
from scipy.signal import welch
import matplotlib.pyplot as plt

# Folder containing the .mat files
data_path = 'data'

# Collect filenames
ictal_files = sorted([f for f in os.listdir(data_path) if 'ictal' in f and f.endswith('.mat')])
interictal_files = sorted([f for f in os.listdir(data_path) if 'interictal' in f and f.endswith('.mat')])

def load_eeg_data(filepath):
    mat = sio.loadmat(filepath)
    data = np.array(mat['data'])
    fs = float(mat['freq'][0][0])  # Sampling frequency
    return data, fs

def compute_avg_psd(files):
    psd_list = []
    for f in files:
        data, fs = load_eeg_data(os.path.join(data_path, f))
        # Average across channels if multichannel
        if data.ndim > 1:
            data = np.mean(data, axis=0)
        f_vals, Pxx = welch(data, fs=fs, nperseg=fs*2)
        psd_list.append(Pxx)
    avg_psd = np.mean(psd_list, axis=0)
    return f_vals, avg_psd

# Compute PSD for seizure and baseline
f_ictal, psd_ictal = compute_avg_psd(ictal_files)
f_interictal, psd_interictal = compute_avg_psd(interictal_files)

# Plotting
plt.figure(figsize=(10,6))
plt.semilogy(f_ictal, psd_ictal, label='Seizure (Ictal)', color='r')
plt.semilogy(f_interictal, psd_interictal, label='Baseline (Interictal)', color='b')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (V²/Hz)')
plt.title('Average PSD of Intracranial EEG (Ictal vs Interictal)')
plt.legend()
plt.grid(True)
plt.show()
