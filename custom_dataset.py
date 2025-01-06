from torch.utils.data import Dataset
import pandas as pd
import audio_functions as auf
import os

class GunShotsNoisesDataset(Dataset):

    def __init__(self, metadata_file, audios_dir):
        self.shots_metadata = pd.read_excel(metadata_file, sheet_name="Shots")
        self.noises_metadata = pd.read_excel(metadata_file, sheet_name="Noise")
        self.audios_dir = audios_dir


    def __len__(self):
        """Returns ammount of samples in data"""
        shots_len = len(self.shots_metadata)
        noises_len = len(self.noises_metadata)
        total_len = shots_len + noises_len

        return total_len




    def __getitem__(self, index):
        """Returns item by index"""
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        audio_signal, fs = auf.load_audio(audio_sample_path, output_format="torch")
        return audio_signal, label 

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
    

if __name__ == "__main__":
    metadata_file = "./dataset/metadata.xlsx"
    audios_dir = "./dataset"
    GSN_Dataset = GunShotsNoisesDataset(metadata_file, audios_dir)
    print(len(GSN_Dataset))

    signal_1, label_1 = GSN_Dataset[0]
    signal_2, label_2 = GSN_Dataset[250]

    print(label_1)
    print(label_2)