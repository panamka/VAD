import pandas as pd
import numpy as np

import os
import sys

import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt

import h5py
from pydub import AudioSegment

def create_waveplot(data, sr):
    plt.figure(figsize=(10, 3))
   
    librosa.display.waveplot(data, sr=sr)    
    plt.show()
    
def create_spectrogram(data, sr):
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(12, 3))
    
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')   
    plt.colorbar()
    

def create_M_spect(data, sr, n_mfcc):
    plt.figure(figsize=(10, 3))
    
    mfccs = librosa.feature.mfcc(data, sr=sr, n_mfcc=n_mfcc)
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    
def noise(data):
    noise_factor = 0.035*np.random.uniform()*np.amax(data)
    noise = np.random.normal(size=data.shape[0])
    data = data + noise_factor*noise
    return data




