import pickle

import numpy as np
import torch
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from LSTMModel import LSTMModel

if torch.cuda.is_available():
    print("CUDA is available! PyTorch is using GPU acceleration.")
    device = "cuda:1"
else:
    print("CUDA is not available. PyTorch is using CPU.")
    device = "cpu"

columns_to_look = ['EEG.Fp1',	'EEG.AF3',	'EEG.F3',	'EEG.FC1'	,'EEG.C3'	,'EEG.FC3'	,'EEG.T7','EEG.CP5','EEG.CP1'
                    ,'EEG.P1'	,'EEG.P7'	,'EEG.P9',	'EEG.PO3'	,'EEG.O1'	,'EEG.O9',	'EEG.POz',	'EEG.Oz',	'EEG.O10',	'EEG.O2',
                   'EEG.PO4'	,'EEG.P10',	'EEG.P8',	'EEG.P2','EEG.CP2',	'EEG.CP6'	,'EEG.T8',	'EEG.FC4',	'EEG.C4',
                   'EEG.FC2',	'EEG.F4', 'EEG.AF4',	'EEG.Fp2']

columns_to_look_no_visual = ['EEG.Fp1',	'EEG.AF3',	'EEG.F3','EEG.FC1','EEG.C3','EEG.FC3','EEG.T7',
                   'EEG.T8','EEG.FC4','EEG.C4','EEG.FC2','EEG.F4','EEG.AF4','EEG.Fp2']

input_size = 32  # Number of features (channels)
hidden_size = 128  # Number of LSTM units
num_layers = 4 # Number of LSTM layers
num_classes = 10  # Number of unique labels
batch_size = 1
learning_rate = 0.0001
num_epochs = 30

model = LSTMModel(input_size, hidden_size, num_layers, num_classes, device=device, contrastive=False).to(device)
# Load the trained model
model.load_state_dict(torch.load('lstm_model.pth'))
model.train()

with open("data/forLSTM/X.pck", 'rb') as f:
    X = pickle.load(f)
with open("data/forLSTM/Y.pck", 'rb') as f:
    Y = pickle.load(f)

res_per_channel = []
res_summarize_channel = []
for label in tqdm(np.unique(Y), total=10):
    total_attributions = []
    indices = np.where(Y == label)[0]
    Y_filtered = Y[indices]
    X_filtered = X[indices, :, :]

    X_filtered = torch.tensor(X_filtered, dtype=torch.float32).to(device)
    Y_filtered = torch.tensor(Y_filtered, dtype=torch.long).to(device)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_filtered, Y_filtered),
                                               batch_size=batch_size,
                                               shuffle=True)


    for i, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader),position=0, leave=False):
        integrated_gradients = IntegratedGradients(model)
        attributions, _ = integrated_gradients.attribute(inputs, target=labels, return_convergence_delta=True)
        attributions = attributions.squeeze().cpu().numpy()
        total_attributions.append(np.abs(attributions))

    mean_values = np.mean(total_attributions, axis=0)
    tmp_data = pd.DataFrame(mean_values)
    tmp_data.columns = columns_to_look
    tmp_data['number'] = label
    res_per_channel.append(tmp_data)

    mean_values_mean_values = np.mean(mean_values, axis=0)
    tmp_data_mean = pd.DataFrame([mean_values_mean_values])
    tmp_data_mean.columns = columns_to_look
    tmp_data_mean['number'] = label
    res_summarize_channel.append(tmp_data_mean)

    plt.figure(figsize=(10, 5))
    plt.imshow(mean_values.T, cmap='RdBu', aspect='auto')
    plt.colorbar()
    plt.title('Integrated Gradients {}'.format(label))
    plt.xlabel('Time Step')
    plt.ylabel('Feature Index')
    plt.yticks(np.arange(len(columns_to_look)), columns_to_look)
    plt.show()
    print()
res_per_channel = pd.concat(res_per_channel)
res_per_channel.to_csv("IG_res_per_channel.csv")

res_summarize_channel = pd.concat(res_summarize_channel)
res_summarize_channel.to_csv("IG_res_summarize_channel.csv")