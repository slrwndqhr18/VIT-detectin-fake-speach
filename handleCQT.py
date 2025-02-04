import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from handleMultiPs import Multi_PS_wrapper
def _audio_to_cqt_image(audio_path, output_path = "/content/cqt_image", sr=22050, hop_length=512, fmin=None, n_bins=84, bins_per_octave=12):

    # Load audio file
    y, _ = librosa.load(audio_path, sr=sr)

    # Check input signal length
    min_signal_length = hop_length * (n_bins // bins_per_octave)
    if len(y) < min_signal_length:
        # Pad if input signal is too short
        pad_width = min_signal_length - len(y)
        y = np.pad(y, (0, pad_width), mode='constant')
    n_fft = min(len(y), 256)

    # CQT transfer
    CQT = librosa.cqt(y, sr=sr, hop_length=hop_length, fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave)
    CQT_db = librosa.amplitude_to_db(np.abs(CQT), ref=np.max)

    # Draw CQT image
    plt.figure(figsize=(2.24, 2.24))
    librosa.display.specshow(CQT_db, sr=sr, hop_length=hop_length, x_axis=None, y_axis=None, fmin=fmin)
    plt.axis('off')  # Remove axis from CQT image

    # Save image as file
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

@Multi_PS_wrapper
def Conv_to_CQT_img(_param):
    # Create CQT image for all audio files
    for id in _param["ids"]:
        # Call function audio_to_cqt_image
        filePath = _param["inputPath"] + id + ".ogg"
        _audio_to_cqt_image(filePath, output_path=os.path.join(_param["outputPath"], f"{id}_cqt.png"))
    return _param["i"]
