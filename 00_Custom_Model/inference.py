import torch
import torchaudio


from cnn import CNNNetwork
from torchsummary import summary
from urbansounddataset import UrbanSoundDataset
from train import AUDIO_DIR, ANNOTATIONS_FILE, SAMPLE_RATE, NUM_SAMPLES , FILE_NAME


class_mapping = [
    "Car",
    "Bike",
    "Bus",
    "Noise"
]


def predict(model, input, target, class_mapping):
    model.eval()                                            # -> every time evaluate // gradients only needs when we actually training
    with torch.no_grad():
        """        
        These are Tensor objects with a specific shape
        Tensor (1, 10) -> [[0.1, 0.01 , ... , 0.6]] if sum all of them it will get 1 point cuz of softmax
        (1,10) 1 = numberof samples that we are passing to the model
              10 = the number of classes that the model tries to predicts ( we have 10 classes digits so 10 )
        so we will catch out highest number of predictions 
        """
        predictions = model(input)                          # -> pass the input to model and get back to predictions
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]       
        predicted_index = predictions[0].argmax(0)          # -> associated with the highest value & we'll take it for axis zero 
        predicted = class_mapping[predicted_index]          # mapping this prediected index to relative class
        expected = class_mapping[target]
    return predicted, expected


if __name__ == "__main__":
    # load back the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} Device...")
    cnn = CNNNetwork().to(device)                           # -> ready to load pytorch model
    state_dict = torch.load(FILE_NAME)
    cnn.load_state_dict(state_dict)                         # -> directly to pytorch  loading back

    # load urban sound dataset dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = UrbanSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)


    # get a sample from the urban sound dataset for inference
    # TF = [num_channels, freq, time] // PyTorch cnn -> [batch size , num_channels , freq , time]
    a = 0 
    while a != "q" :
        m_l = 0
        a,b = map(int,input("Get Range : ").split())
        for i in range(a,b):
        
            input0, target = usd[i][0], usd[i][1] # [batch size, num_channels, fr, time]
            input0.unsqueeze_(0)

            # make an inference
            predicted, expected = predict(cnn, input0, target,class_mapping)
            if predicted == expected : 
                print(f"File Path : {usd._get_audio_sample_path(i)}\t{expected}")
                m_l +=1 
            if i % 100 == 0 :
                print(f"\n---------{i+1-a} Files Passed---------\n")

        if m_l % 100 == 0 :
            print(f"\n---------Matched Files {m_l}/{b-a}---------\n")
        
        print(f"=============== All Matched Files {m_l}/{b-a} ====================")        

            # print(f"Predicted: '{predicted}', expected: '{expected}'")

    # print(type(usd[0]),usd[0],sep="\n")
    # print(type(usd[0][0].unsqueeze_(0)),usd[0][0].unsqueeze_(0),sep="\n")

    # input1, target1 = usd[0][0], usd[0][1]
    # input1.unsqueeze_(0)

    # predicted1, expected1 = predict(cnn,input1,target1,class_mapping)
    # print(f"Predicted: '{predicted1}', expected: '{expected1}'")
    