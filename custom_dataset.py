from torch.utils.data import Dataset
import pandas as pd
import audio_functions as auf
import os
import pywt
import torch

class GunShotsNoisesDataset(Dataset):

    def __init__(self, metadata_file, audios_dir, transformation, sample_rate):
        """
        Instance constructor
        Input:
            - metadata_file: str type object. metadata file path.
            - audios_dir: str type object. Audios folder path
            - transformation: dict type object for cwt transformation. Values:
                - wavelet
                - scales
        """
        self.shots_metadata = pd.read_excel(metadata_file, sheet_name="Shots")
        self.noises_metadata = pd.read_excel(metadata_file, sheet_name="Noise")
        self.audios_dir = audios_dir
        self.transformation_configure = transformation
        self.sample_rate = sample_rate

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
        transformed_signal = self._apply_transformation(audio_signal)

        return audio_signal, transformed_signal, label 

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
            audio_signal = audio_signal.numpy()
    
        wavelet = self.transformation_configure["wavelet"]
        scales = self.transformation_configure["scales"]
    
        # Apply CWT
        coeffs, freqs = pywt.cwt(audio_signal, scales, wavelet, sampling_period=1/self.sample_rate)
    
        # Convert the coefficients back to a PyTorch tensor
        coeffs = torch.tensor(coeffs, dtype=torch.float32)
    
        return coeffs


if __name__ == "__main__":
    metadata_file = "./dataset/metadata.xlsx"
    audios_dir = "./dataset"



    GSN_Dataset = GunShotsNoisesDataset(metadata_file, audios_dir)

