import os
import torch
from torch.utils.data import Dataset

from dataset.preprocess import get_features_for_chordformer, preprocess_data_for_chordformer


class GuitarChordDataset(Dataset):
    def __init__(self, root_dir, chord_label_mapping, sample_rate=16000, n_fft=1024, hop_length=512, n_mels=64):
        """
        Initialises the GuitarChordDataset.

        Parameters:
        - root_dir (str): The root directory containing audio files.
        - chord_label_mapping (dict): A mapping from chord names to label indices.
        - sample_rate (int): The sample rate for audio processing.
        - n_fft (int): Number of FFT components for the Mel spectrogram.
        - hop_length (int): The hop length for the Mel spectrogram.
        - n_mels (int): The number of Mel bands.
        """
        self.root_dir = root_dir
        self.features = []  # hold features and labels
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.chord_label_mapping = chord_label_mapping

        # traversing through each chord directory in the root directory
        for chord_dir in os.listdir(root_dir):
            chord_path = os.path.join(root_dir, chord_dir)
            if os.path.isdir(chord_path):  # if the path is a directory
                for file in os.listdir(chord_path):
                    if file.endswith('.wav'):  # process .wav files only
                        file_path = os.path.join(chord_path, file)
                        # extracting Mel spectrogram and file name
                        mel, file_name = get_features_for_chordformer(
                            file_path, self.sample_rate, self.n_fft, self.hop_length, self.n_mels
                        )
                        # parse labels from the file name
                        label_info = self._parse_labels(file)
                        # append features and labels to the list
                        self.features.append((mel, chord_dir, label_info))

        # preprocess the data and convert to tensors
        self.X_tensor, self.y_tensor = preprocess_data_for_chordformer(self.features)

    def _parse_labels(self, file_name):
        """
        Parses the labels from the file name.

        Parameters:
        - file_name (str): The name of the audio file.

        Returns:
        - list: A list containing the chord label, style label, guitar type label, and guitar status label.
        """
        # splitting the file name to extract information
        parts = file_name.split('_')
        chord = parts[0]  # chord name
        guitar_type = parts[1]  # guitar type
        guitar_status = parts[2]  # guitar status (Synthetic or Real)
        style_info = parts[3].split('.')[0]  # extract style information
        style = int(style_info.split('_')[-1])  # converting style to an integer

        # map the chord name to a label
        chord_label = self.chord_label_mapping[chord]
        # encoding the guitar type as a label
        guitar_type_label = self._encode_guitar_type(guitar_type)
        # determining if the guitar status is Synthetic (0) or Real (1)
        guitar_status_label = 0 if 'Synthetic' in guitar_status else 1
        # converting style to a zero-indexed label
        style_label = style - 1

        return [chord_label, style_label, guitar_type_label, guitar_status_label]

    def _encode_guitar_type(self, guitar_type):
        """
        Encodes the guitar type as an integer label.

        Parameters:
        - guitar_type (str): The type of guitar (e.g., 'Acoustic').

        Returns:
        - int: The index of the guitar type in the list.
        """
        types = ['Acoustic', 'Classical', 'Electric']
        return types.index(guitar_type)

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
        - int: The number of samples.
        """
        return len(self.X_tensor)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset.

        Parameters:
        - idx (int): The index of the sample to retrieve.

        Returns:
        - tuple: A tuple containing the Mel spectrogram tensor and the label tensor.
        """
        # converting the Mel spectrogram and labels to PyTorch tensors
        return torch.tensor(self.X_tensor[idx], dtype=torch.float32), torch.tensor(self.y_tensor[idx], dtype=torch.long)
