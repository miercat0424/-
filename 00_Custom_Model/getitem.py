import os
import torch
import pandas as pd
import torchaudio


from torch.utils.data import Dataset

def print_stats(waveform, sample_rate=None, src=None):
    if src:
        print("-" * 10)
        print("Source:", src)
        print("-" * 10)
    if sample_rate:
        print("Sample Rate:", sample_rate)
    print("Shape:", tuple(waveform.shape))
    print("Dtype:", waveform.dtype)
    print(f" - Max:     {waveform.max().item():6.3f}")
    print(f" - Min:     {waveform.min().item():6.3f}")
    print(f" - Mean:    {waveform.mean().item():6.3f}")
    print(f" - Std Dev: {waveform.std().item():6.3f}")

class UrbanSoundDataset(Dataset):

    def __init__(self,                  # -> annotations_file = files that all annotations (strings 주석)
                 annotations_file,
                 audio_dir,             
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device):  
        self.annotations = pd.read_csv(annotations_file)    # -> load csv pandas dataFrame
        self.audio_dir = audio_dir                          # -> audio_dir = path to where we store the audio samples
        self.device = device
        self.transformation = transformation.to(self.device)# -> using GPU it will transform wav files to mel Spectrogram
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        # -> want to return number of datasets
        return len(self.annotations)

    def __getitem__(self, index):
        # -> getting loading wavefrom audio samples , at the same time return to label
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        # print_stats(signal,sample_rate=self.target_sample_rate)
        # -> torchaudio loading the data // each OS have differ function // torchaudio.load , torchaudio.load_wav and torchaudio.save
        # signal -> pytorch 2Dimesion tensor // (num_channels , sampls) -> (2,16000) -> (1,16000) // 2 = stereo
        signal = signal.to(self.device)                     # -> before resampling use CUDA
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)        # -> trying to make mono
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)                # -> mel_spectrogram passing the signal
        return signal, label

    def _cut_if_necessary(self, signal):
        # -> if the signal has more samples as we expected than cut it // signal -> Tensor -> (1, num_samples)
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]           # -> (1,50k) -> expected NUM_SAMPLES = (1, 22050)
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]                     # -> expand our samples if that is shorter than what we expected (NUM_SAMPLE)
        if length_signal < self.num_samples:                # [1,1,1] -> [1,1,1,0,0] expand on the right side // left side = pre pan
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
            # last_dim_padding = (0,2) -> // [1,1,1] -> [1,1,1,0,0]
            # last_dim_padding = (1,2) -> // [1,1,1] -> [0,1,1,1,0,0]
            # last_dim_padding = (1,1,2,2) -> // (1, num_samples)
        return signal

    def _resample_if_necessary(self, signal, sr):           # -> already have in torchaudio but wanna configure 
        if sr != self.target_sample_rate:                   # -> if the sample rate has differ then change it to all the same
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device) # resmapler gets Tensor Data to Device
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):               # -> if signal is stereo than change it to mono
        if signal.shape[0] > 1:                             # -> signal.shape[0] = number of channels
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        print(f"index : {index}")
        fold = f"fold{self.annotations.iloc[index, 5]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[
            index, 0])                                      # -> find a wav file 
        print(path)
        return path

    def _get_audio_sample_label(self, index):               # -> take a label information
        return self.annotations.iloc[index, 6]

    # what item is useful // a_list[1] -> a_list.__getitem__(1)

if __name__ == "__main__":
    ANNOTATIONS_FILE = "D:/PyAud/pytorchforaudio/UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR = "D:/PyAud/pytorchforaudio/UrbanSound8K/audio"   
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050                                     # -> one second of work audio

    device = "cpu"
    # if torch.cuda.is_available():
    #     device = "cuda"
    # else:
    #     device = "cpu"
    # print(f"Using device {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram( # -> can transform what you want 
        sample_rate=SAMPLE_RATE,
        n_fft=1024,                                         # -> frame size
        hop_length=512,                                     # -> set it the half of n_fft           
        n_mels=64
    )

    usd = UrbanSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)
    print(f"There are {len(usd)} samples in the dataset.")
    signal, label = usd[0]
    print(signal,label,sep="\n")
    


    # -> signal shape (1 = channels , 64 = n_mels , 10 = number of frame )


