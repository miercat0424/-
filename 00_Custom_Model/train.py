import torch
import torchaudio

from torch import nn
from torchsummary import summary
from torch.utils.data import DataLoader
from urbansounddataset import UrbanSoundDataset
from cnn import CNNNetwork

# 1- download dataset
# 2- create data loader
# 3- build model            <- CNN Network
# 4- train
# 5- save trained model
# ---------------------------------------------------------------------------------

FILE_NAME = "SM-Instrument-211009-03-SoftMax"
BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.0001

ANNOTATIONS_FILE = "D:\PyAud\SM-Instruments\Test2.csv"
AUDIO_DIR = "D:\PyAud\SM-Instruments\WAV_Files"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050

def create_data_loader(train_data, batch_size):    
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    # print(train_dataloader) # -> shape error 시에 사용
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):     # -> what training model
    # loop for all samples  // batch access and wide
    for input, target in data_loader:                                       # -> sets cal , backpropagate loss , weight each batchs
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input).to(device)                                # -> need to get prediction
        loss = loss_fn(prediction, target)                                  # -> expected values and prediction

        # backpropagate error and update weights
        optimiser.zero_grad()                                               # -> cal gradient to decide how to update // each iteration gets to 0 gradients
        loss.backward()
        optimiser.step()                                                    # -> updating weights // optimiser is the important thing on weighting

    print(f"loss: {loss.item()}")                                           # -> printing the loss for last batch that have                                 


def train(model, data_loader, loss_fn, optimiser, device, epochs):          # -> each iteration 
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")


if __name__ == "__main__":
    # 3 builds model (excute)---------------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # instantiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    ).to(device)
    usd = UrbanSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)
    train_dataloader = create_data_loader(usd, BATCH_SIZE)
    # construct model and assign it to device
    cnn = CNNNetwork().to(device)
    # print(cnn)
    # 4 instantiate loss function + optimiser ---------------------------------------------------------------------------------
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(cnn, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(cnn.state_dict(), f"{FILE_NAME}")
    summary(cnn.to(device), (1,64, 44))
    print(f"Trained feed forward net saved at {FILE_NAME}")