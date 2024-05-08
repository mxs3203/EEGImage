import pickle
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import random_split

import wandb
from Discriminator import Discriminator
from ImageGenerationDataset import ImageGenerationDataset
from ImageGeneratorModel import ImageGenerator
from torch import optim

from LSTMModel import LSTMModel

if torch.cuda.is_available():
    print("CUDA is available! PyTorch is using GPU acceleration.")
    device = "cuda:1"
else:
    print("CUDA is not available. PyTorch is using CPU.")
    device = "cpu"


with open("data/forLSTM/X.pck", 'rb') as f:
    X = pickle.load(f)
with open("data/forLSTM/Y.pck", 'rb') as f:
    Y = pickle.load(f)

print(np.shape(X), np.shape(Y))

FINE_TUNE = False
GAN = False
input_size = 32  # Number of features (channels)
hidden_size = 128  # Number of LSTM units
num_layers = 4 # Number of LSTM layers
batch_size = 256
learning_rate = 0.0001
num_epochs = 50
contrastive_output_size = 64
# Create the LSTM autoencoder model
model = LSTMModel(input_size, hidden_size, num_layers, contrastive_output_size, device=device, contrastive=True).to(device)
model.load_state_dict(torch.load("lstm_contrsative_model_64.pth"))
if not FINE_TUNE:
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

image_generatin_dataset = ImageGenerationDataset(data=X, targets=Y,
                                                 mnist_folder="data/MNIST",
                                                 device=device,
                                                 number_of_imgs_per_class=1000)
train_size = int(0.8 * len(image_generatin_dataset))  # 80% for training
test_size = len(image_generatin_dataset) - train_size
train_dataset, test_dataset = random_split(image_generatin_dataset,
                                           [train_size, test_size]
                                           )
image_generatin_dataset_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
image_generatin_dataset_valid_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

image_generator = ImageGenerator(device=device, image_channels=1, input_size=contrastive_output_size, scale_factor=7).to(device)
optimizer = optim.Adam(image_generator.parameters(), lr=learning_rate)
if FINE_TUNE:
    optimizer2 = optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.BCEWithLogitsLoss()
run = wandb.init(
    # set the wandb project where this run will be logged
    project="EEGImage",
    name="ImageGenerator-CNN-{}".format(contrastive_output_size),

    config={
        "learning_rate": learning_rate,
        "architecture": "LSTM_Contrastive_ImageGeneration",
        "dataset": "EEG",
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_epochs": num_epochs,
        "output_size": contrastive_output_size,
        "fine_tune": FINE_TUNE
    }
)


# Training loop
total_steps = len(image_generatin_dataset_train_loader)
for epoch in range(num_epochs):
    train_losses = []
    val_losses = []
    train_losses_D = []
    val_losses_D = []
    if FINE_TUNE:
        model.train()
    image_generator.train()
    for i, (x,y,images) in enumerate(image_generatin_dataset_train_loader):
        optimizer.zero_grad()
        if FINE_TUNE:
            optimizer2.zero_grad()

        extracted_eeg_features = model(x)
        generated_image = image_generator(extracted_eeg_features)
        loss = criterion(generated_image, images)
        train_losses.append(loss.item())
        # Backward and optimize
        loss.backward()
        optimizer.step()


        if FINE_TUNE:
            optimizer2.step()
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_steps}], Loss: {loss.item():.4f} ')

    # Validation
    if FINE_TUNE:
        model.eval()
    image_generator.eval()
    with torch.no_grad():
        all_preds = []
        for i, (x,y,images) in enumerate(image_generatin_dataset_valid_loader):
            extracted_eeg_features = model(x)
            generated_image = image_generator(extracted_eeg_features)
            loss = criterion(generated_image, images)
            val_losses.append(loss.item())

        if epoch % 5 == 0: # every 5th epoch show generated image
            fig, axes = plt.subplots(3, 3, figsize=(10, 10))
            for i in range(9):
                random_image = generated_image.detach().cpu().numpy()[i, 0, :, :]
                random_image_label = y.detach().cpu().numpy()[i]
                row, col = i // 3, i % 3
                axes[row, col].imshow(random_image, cmap='gray')
                axes[row, col].set_title(f'Label: {random_image_label}')
                axes[row, col].axis('off')
            run.log({"GenerateImage": fig})

    run.log({"Train/Loss": np.mean(train_losses), "Valid/Loss": np.mean(val_losses)})
torch.save(model.state_dict(), 'image_generation_model_CNN.pth')
run.log_model('image_generation_model.pth', "ImageGenerationModel")
wandb.finish()

print('Training finished.')