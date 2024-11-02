import numpy as np
import torch
import torchaudio


# gen a Mel spectrogram from an audio waveform
def get_melspectrogram(waveform, sr, n_fft, hop_length, n_mels):
    # MelSpectrogram transform with the specified parameters
    melspec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    # apply the MelSpectrogram transform to the waveform
    melspec = melspec_transform(waveform)

    # converting the power spectrogram to decibel units for better visualisation and feature scaling
    melspec_db = torchaudio.transforms.AmplitudeToDB()(melspec)

    # single channel (for CNN input) and return it as a np array
    return melspec_db.numpy().reshape((n_mels, -1, 1))


# extract features (Mel spectrogram) from an audio file
def get_features_for_chordformer(file, sr, n_fft, hop_length, n_mels):
    # audio file using torchaudio
    waveform, sample_rate = torchaudio.load(file)

    # converting stereo audio to mono by averaging the channels if necessary
    waveform = waveform.mean(dim=0)

    # gen the Mel spectrogram for the audio waveform
    mel = get_melspectrogram(waveform, sr, n_fft, hop_length, n_mels)

    return mel, file


# preprocess data and prepare it for training
def preprocess_data_for_chordformer(features):
    # init lists to store the processed spectrograms and corresponding labels
    X, y = [], []

    # iterate over each feature entry
    for feat in features:
        mel = feat[0]  # the Mel spectrogram
        chord_label, style_label, guitar_type_label, guitar_status_label = feat[2]  # labels

        # spectrogram array filled with a low decibel value (-80 dB)
        tmp_spec = np.full((64, 128, 1), -80, dtype=np.float32)

        # number of time steps to copy
        time_steps = min(mel.shape[1], 128)

        # copying the Mel spectrogram data into the temporary array
        tmp_spec[:, :time_steps, :] = mel[:, :time_steps, :]

        # append the processed spectrogram and labels to the lists
        X.append(tmp_spec)
        y.append([chord_label, style_label, guitar_type_label, guitar_status_label])

    # convert the lists to np arrays
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    # processed spectrograms and labels
    return X, y
