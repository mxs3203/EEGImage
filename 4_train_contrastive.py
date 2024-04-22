
import pickle
import pandas as pd
import numpy as np
import torch
import umap
from matplotlib import pyplot as plt
import seaborn as sb
import wandb
from sklearn.metrics.pairwise import cosine_similarity

from losses import SupConLoss
from sklearn.model_selection import train_test_split
from torch import optim

from LSTMModel import LSTMModel

if torch.cuda.is_available():
    print("CUDA is available! PyTorch is using GPU acceleration.")
    device = "cuda:1"
else:
    print("CUDA is not available. PyTorch is using CPU.")
    device = "cpu"

for output_size_ in [32,64,128,256,512]:
    for bs in [2048]:
        with open("data/forLSTM/X.pck", 'rb') as f:
            X = pickle.load(f)
        with open("data/forLSTM/Y.pck", 'rb') as f:
            Y = pickle.load(f)

        print(np.shape(X), np.shape(Y))

        input_size = 32  # Number of features (channels)
        hidden_size = 128  # Number of LSTM units
        num_layers = 4 # Number of LSTM layers
        batch_size = bs
        learning_rate = 0.0001
        num_epochs = 400
        temp = 0.03
        base_temperature = 0.08
        output_size = output_size_
        # Create the LSTM autoencoder model
        model = LSTMModel(input_size, hidden_size, num_layers, output_size, device=device, contrastive=True).to(device)

        # Define loss function and optimizer
        criterion = SupConLoss(temperature=temp, base_temperature=base_temperature, device=device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        def intra_cluster_similarity(embeddings, labels):
            similarity_scores = []
            unique_labels = np.unique(labels)
            for label in unique_labels:
                indices = np.where(labels == label)[0]
                cluster_embeddings = embeddings[indices]
                similarity_matrix = cosine_similarity(cluster_embeddings)
                np.fill_diagonal(similarity_matrix, 0)  # Exclude self-similarity
                intra_similarity = np.mean(similarity_matrix)
                similarity_scores.append(intra_similarity)
            return np.mean(similarity_scores)

        def inter_cluster_similarity(embeddings, labels):
            similarity_matrix = cosine_similarity(embeddings)
            unique_labels = np.unique(labels)
            inter_similarity_scores = []
            for label1 in unique_labels:
                for label2 in unique_labels:
                    if label1 != label2:
                        indices1 = np.where(labels == label1)[0]
                        indices2 = np.where(labels == label2)[0]
                        inter_similarity = np.mean(similarity_matrix[np.ix_(indices1, indices2)])
                        inter_similarity_scores.append(inter_similarity)
            return np.mean(inter_similarity_scores)

        run = wandb.init(
            # set the wandb project where this run will be logged
            project="EEGImage",
            name='Contrastive-{}-{}'.format(output_size, batch_size),

            config={
                "learning_rate": learning_rate,
                "architecture": "LSTM_Contrastive",
                "dataset": "EEG",
                "batch_size": batch_size,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "num_epochs": num_epochs,
                "temp":temp,
                "base_temperature": base_temperature,
                "output_size": output_size

            }
        )



        X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.3)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
        # Data loader
        train_data = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

        val_data = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

        # Training loop
        total_steps = len(train_loader)
        for epoch in range(num_epochs):
            model.train()
            train_losses = []
            val_losses = []
            for i, (inputs, labels) in enumerate(train_loader):
                # Forward pass
                outputs = model(inputs)
                outputs = outputs.unsqueeze(dim=1)
                loss = criterion(outputs, labels)
                train_losses.append(loss.item())
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (i + 1) % 100 == 0:
                   print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_steps}], Loss: {loss.item():.4f}')

            # Validation
            model.eval()
            with torch.no_grad():
                all_preds = []
                all_labels = []
                all_features = []
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    all_features.append(pd.DataFrame(outputs.cpu().detach().numpy()))
                    all_labels.append(labels.cpu().detach().numpy())
                    outputs = outputs.unsqueeze(dim=1)
                    loss = criterion(outputs, labels)
                    val_losses.append(loss.item())
                if epoch % 5 == 0: # every 5th epoch make umap
                    print("Making a UMAP...")
                    umap_data = pd.concat(all_features, ignore_index=True)
                    umap_data['label'] = np.concatenate(all_labels)

                    umap_data = umap_data.sample(5000)
                    labels = umap_data['label'].values

                    intra = intra_cluster_similarity(umap_data.drop(columns=['label']).values, labels)
                    inter = inter_cluster_similarity(umap_data.drop(columns=['label']).values, labels)
                    run.log({"Valid/IntraClusterSimilarity": intra, "Valid/InterClusterSimilarity": inter, "Valid/Cohesion": intra-inter})
                    reducer = umap.UMAP()
                    embedding = pd.DataFrame(reducer.fit_transform(umap_data.drop(columns=['label'])))
                    embedding['label'] = labels
                    embedding['label'] = embedding['label'].astype("category")
                    embedding.columns = ['V0', 'V1', 'label']
                    plt.scatter(embedding['V0'], embedding['V1'], c=embedding['label'], cmap='tab10')
                    plt.title('UMAP Visualization Epoch-{}'.format(epoch))
                    plt.xlabel('UMAP Dimension 1')
                    plt.ylabel('UMAP Dimension 2')
                    plt.legend()
                    run.log({"UMAP": plt})

                    if epoch > 380:
                        reducer = umap.UMAP()
                        embedding = pd.DataFrame(reducer.fit_transform(umap_data.drop(columns=['label'])))
                        embedding['label'] = labels
                        embedding['label'] = embedding['label'].astype("category")
                        embedding.columns = ['V0', 'V1', 'label']
                        sb.scatterplot(embedding, x = 'V0', y='V1', hue='label')
                        plt.xlabel('UMAP Dimension 1')
                        plt.ylabel('UMAP Dimension 2')
                        plt.savefig('plots/Contrastive-{}_{}.svg'.format(output_size, epoch), format = 'svg')
                        plt.close()

            run.log({"Train/Loss": np.mean(train_losses), "Valid/Loss": np.mean(val_losses)})
        torch.save(model.state_dict(), 'lstm_contrsative_model_{}.pth'.format(output_size))
        run.log_model('lstm_contrsative_model_{}.pth'.format(output_size), "ContrastiveModelLSTM")
        wandb.finish()

    print('Training finished.')
