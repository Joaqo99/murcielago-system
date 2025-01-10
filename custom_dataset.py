from torch.utils.data import Dataset
import pandas as pd
import audio_functions as auf
import os
import pywt
import torch
from torchaudio import transforms
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

class GunShotsNoisesDataset(Dataset):

    def __init__(self, metadata_file, audios_dir, transformation, sample_rate, audio_duration = None):
        """
        Instance constructor
        Input:
            - metadata_file: str type object. metadata file path.
            - audios_dir: str type object. Audios folder path
            - transformation: dict type object for cwt transformation. Values:
                - wavelet
                - scales
            - sample_rate: int type object. Sample rate to work with.

        """
        self.shots_metadata = pd.read_excel(metadata_file, sheet_name="Shots")
        self.noises_metadata = pd.read_excel(metadata_file, sheet_name="Noise")
        self.audios_dir = audios_dir
        self.transformation_configure = transformation
        self.sample_rate = sample_rate

        if audio_duration:
            self.N_samples = int(audio_duration * sample_rate)
            self.return_waveform = True
        else:
            self.N_samples = self.transformation_configure["scales"]
            self.return_waveform = False

    def __len__(self):
        """Returns ammount of samples in data"""
        shots_len = len(self.shots_metadata)
        noises_len = len(self.noises_metadata)
        total_len = shots_len + noises_len

        return total_len

    def __getitem__(self, index):
        """
        Returns item and its label by index. The signal is expressed both in waveform and the cwt transform coefficients.
        Input:
            - index: int type object.
        Output:
            - audio_signal: torch tensor vector type object. Audio waveform
            - transformed signal: tuple type object. Audio spectrogram
            - label: str type object.
        """

        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        audio_signal, fs = auf.load_audio(audio_sample_path)
        audio_signal = auf.to_mono(audio_signal)
        audio_signal = auf.resample_signal_fs(audio_signal, fs, self.sample_rate)
        event_signal = self.detect_event(audio_signal, 0.0001)

        transformed_signal = self._apply_transformation(event_signal)
        if self.return_waveform:
            return event_signal, transformed_signal, label
        else:
            return transformed_signal, label

    def _get_audio_sample_path(self, index):

        if index < len(self.shots_metadata):
            folder = "Shots"
            audio_file_name = self.shots_metadata.iloc[index]["file name"]

        else:
            folder = "Noises"
            index = index - len(self.shots_metadata)
            audio_file_name = self.noises_metadata.iloc[index]["filename"]

        audio_path = os.path.join(self.audios_dir, folder, audio_file_name)
        return audio_path
        
    def _get_audio_sample_label(self, index):
        if index < len(self.shots_metadata):
            label = "Shot"
        else:
            label = "Noise"

        return label
    
    def detect_event(self, audio_signal, umbral):
        """
        Searchs for impulsive components in audio signal. If no, cuts the signal where dB SPL is present.
        """

        audio_power = audio_signal**2
        audio_power_grad = torch.gradient(audio_power)[0]

        init_interval = int(0.0005*self.sample_rate)
        end_interval = self.N_samples - init_interval

        if torch.max(audio_power_grad) > umbral:
            max_index = audio_power_grad.argmax()
            event_signal = audio_signal[max_index - init_interval: max_index + end_interval]

            return event_signal
        else:
            dB_transform = transforms.AmplitudeToDB(stype="amplitude", top_db=80)
            audio_dB = dB_transform(audio_signal)
            max_dB = torch.argmax(audio_dB)
            event_signal = audio_signal[max_dB - init_interval: max_dB + end_interval]
            return event_signal

    def _apply_transformation(self, audio_signal):
        """
        Applies a continuous wavelet transform (CWT) to the audio signal.
        Input:
            - audio_signal: torch tensor or numpy array, representing the audio waveform.
        Output:
            - coeffs: torch tensor containing the CWT coefficients.
        """
        
        # Convert to numpy if audio_signal is a tensor
        if isinstance(audio_signal, torch.Tensor):
            audio_signal = audio_signal.numpy().astype(np.float32)

        wavelet = self.transformation_configure["wavelet"]
        scales = self.transformation_configure["scales"]

        # Apply CWT
        coeffs, freqs = pywt.cwt(audio_signal, scales, wavelet, sampling_period=1/self.sample_rate)

        # Convert the coefficients back to a PyTorch tensor
        coeffs = torch.tensor(coeffs, dtype=torch.complex32)

        coeffs = coeffs[:,:len(scales)]
        return coeffs
    
def split_dataset(dataset, test_size=0.1, random_seed=42):
    """
    Splits the dataset into training and testing subsets.
    Maintains category proportions (Shots and Noises).
    """
    # Get indices for each category
    shots_indices = list(range(len(dataset.shots_metadata)))
    noises_indices = list(range(len(dataset.shots_metadata), len(dataset)))

    # Split indices for each category
    shots_train_idx, shots_test_idx = train_test_split(
        shots_indices, test_size=test_size, random_state=random_seed
    )
    noises_train_idx, noises_test_idx = train_test_split(
        noises_indices, test_size=test_size, random_state=random_seed
    )

    # Combine indices
    train_indices = shots_train_idx + noises_train_idx
    test_indices = shots_test_idx + noises_test_idx

    return train_indices, test_indices
