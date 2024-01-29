import librosa
import numpy as np
from glob import glob

class SpeechEmotionPreprocessor:
    def __init__(self, sample_rate=22050, max_pad_length=None):
        self.sample_rate = sample_rate
        self.max_pad_length = max_pad_length

    def load_and_pad_audio(self, file_path):
        # Load audio file
        audio, sr = librosa.load(file_path, sr=self.sample_rate)

        # Pad or truncate audio file
        if self.max_pad_length is not None:
            if len(audio) > self.max_pad_length:
                audio = audio[:self.max_pad_length]
            else:
                padding = self.max_pad_length - len(audio)
                audio = np.pad(audio, (0, padding), 'constant')

        return audio

    def load_audios_with_padding(self, dir, max_pad_length=None):
      raws = []
      files = glob(dir+'/*.wav')
      for f in files:
        raw = self.load_and_pad_audio(f, max_pad_length)
        raws.append(raw)

      return np.stack(raws)
      
    def mel_extract(raws, max_size, sr=44100, n_mels=64):
      mels = []

      for r in raws:
        y = librosa.util.fix_length(r, size=max_size, axis=0)
        S = librosa.feature.melspectrogram(y=np.squeeze(y), sr=sr, n_mels=n_mels, n_fft=512)
        S_DB = librosa.power_to_db(S, ref=np.max)
        mels.append(S_DB)

      mels = np.stack(mels)
      return mels

# Example usage:
# preprocessor = SpeechEmotionPreprocessor(max_pad_length=22050 * 3)  # For 3-second audio clips
# audio = preprocessor.load_and_pad_audio("path/to/audio.wav")
# features = preprocessor.extract_features(audio)
