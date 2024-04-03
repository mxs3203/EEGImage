import pickle

import numpy as np
import torch
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
from tqdm import tqdm

from LSTMModel import LSTMModel


columns_to_look = ['EEG.Fp1',	'EEG.AF3',	'EEG.F3',	'EEG.FC1'	,'EEG.C3'	,'EEG.FC3'	,'EEG.T7'	,'EEG.CP5',	'EEG.CP1'	,'EEG.P1'	,'EEG.P7'	,'EEG.P9',	'EEG.PO3'	,'EEG.O1'	,'EEG.O9',	'EEG.POz',	'EEG.Oz',	'EEG.O10',	'EEG.O2',	'EEG.PO4'	,'EEG.P10',	'EEG.P8',	'EEG.P2','EEG.CP2',	'EEG.CP6'	,'EEG.T8',	'EEG.FC4',	'EEG.C4',	'EEG.FC2',	'EEG.F4', 'EEG.AF4',	'EEG.Fp2']

input_size = 32  # Number of features (channels)
hidden_size = 128  # Number of LSTM units
num_layers = 1  # Number of LSTM layers
num_classes = 10  # Number of unique labels
batch_size = 32
learning_rate = 0.0001
num_epochs = 20

model = LSTMModel(input_size, hidden_size, num_layers, num_classes).to("cpu")
# Load the trained model
model.load_state_dict(torch.load('lstm_model.pth'))
model.eval()

with open("data/forLSTM/X.pck", 'rb') as f:
    X = pickle.load(f)
with open("data/forLSTM/Y.pck", 'rb') as f:
    Y = pickle.load(f)


for label in tqdm(np.unique(Y), total=10):
    total_attributions = []
    indices = np.where(Y == label)[0]
    Y_filtered = Y[indices]
    X_filtered = X[indices, :, :]

    X_filtered = torch.tensor(X_filtered, dtype=torch.float32).to("cpu")
    Y_filtered = torch.tensor(Y_filtered, dtype=torch.long).to("cpu")
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_filtered, Y_filtered),
                                               batch_size=1,
                                               shuffle=True)



    for i, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        integrated_gradients = IntegratedGradients(model)
        attributions, _ = integrated_gradients.attribute(inputs, target=labels, return_convergence_delta=True)
        attributions = attributions.squeeze().cpu().numpy()
        total_attributions.append(np.abs(attributions))

    mean_values = np.mean(total_attributions, axis=0)


    plt.figure(figsize=(10, 5))
    plt.imshow(mean_values.T, cmap='RdBu', aspect='auto', origin='lower')
    plt.colorbar()
    plt.title('Integrated Gradients {}'.format(label))
    plt.xlabel('Time Step')
    plt.ylabel('Feature Index')
    plt.yticks(np.arange(len(columns_to_look)), columns_to_look)
    plt.show()
    print()