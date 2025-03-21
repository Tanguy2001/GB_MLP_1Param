import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn
import numpy as np
import shutil
import os
import h5py


class WaveformDataset(Dataset):

    def __init__(self, parameters, waveforms, alpha=0):
        self.parameters = parameters
        self.waveforms = waveforms
        self.alpha = alpha

    def __len__(self):
        return len(self.parameters)

    def __getitem__(self, idx):
        params = self.parameters[idx]
        signal = self.waveforms[idx]

        # Add unit normal noise to the signal
        noise = self.alpha * np.random.normal(size=signal.shape)
        data = signal + noise

        return torch.tensor(data, dtype=torch.float32), torch.tensor(
            params, dtype=torch.float32
        )


"""
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], output_size)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
"""


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        hidden_net_list = []
        hidden_net_list.append(nn.Linear(input_size, hidden_size[0]))

        for i in range(1, len(hidden_size)):
            hidden_net_list.append(nn.Linear(hidden_size[i - 1], hidden_size[i]))

        self.hidden_net_list = nn.ModuleList(hidden_net_list)

        self.fc = nn.Linear(hidden_size[-1], output_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        h = x
        for layer in self.hidden_net_list:
            h = self.relu(layer(h))
        h = self.fc(h)
        return h


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_loop(dataloader, model):
    size = len(dataloader.dataset)
    test_loss = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for X, y in dataloader:
            predictions = model(X)
            loss = criterion(predictions, y)
            test_loss += loss.sum()
    test_loss /= size
    print(f"Test loss: {test_loss:>8f} \n")
    return test_loss


def train_loop(dataloader, model, optimizer):
    size = len(dataloader.dataset)
    train_loss = 0
    criterion = nn.MSELoss()
    for batch, (X, y) in enumerate(dataloader):
        predictions = model(X)

        loss = criterion(predictions, y)

        train_loss += loss.detach().sum()
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 390 == 0:
            loss_value, current = loss.item(), batch * len(X)
            print(f"Loss: {loss_value:>7f}  [{current:>5d}/{size:>5d} samples]")

    average_loss = train_loss.item() / size
    print("Average loss: {:.4f}".format(average_loss))
    return average_loss


def enregistrer_donnees(
    dossier,
    Nom_Waveforms_h5,
    waveforms,
    fichier_yaml,
    Nom_Parametres_h5,
    parameters_standardized,
    Nom_MeanVar_h5,
    MV,
    nom_dossier,
):

    if not os.path.exists(dossier):
        os.makedirs(dossier)

    nom_sous_dossier = os.path.splitext(nom_dossier)[0]
    chemin_sous_dossier = os.path.join(dossier, nom_sous_dossier)

    if not os.path.exists(chemin_sous_dossier):
        os.makedirs(chemin_sous_dossier)

    chemin_complet_h5_waveforms = os.path.join(chemin_sous_dossier, Nom_Waveforms_h5)
    with h5py.File(chemin_complet_h5_waveforms, "w") as hf1:
        hf1.create_dataset(
            "Waveform", data=waveforms, compression="gzip", compression_opts=9
        )

    chemin_complet_h5_parameters = os.path.join(chemin_sous_dossier, Nom_Parametres_h5)
    with h5py.File(chemin_complet_h5_parameters, "w") as hf2:
        hf2.create_dataset(
            "Parameters_standardized",
            data=parameters_standardized,
            compression="gzip",
            compression_opts=9,
        )

    chemin_complet_h5_meanvar = os.path.join(chemin_sous_dossier, Nom_MeanVar_h5)
    with h5py.File(chemin_complet_h5_meanvar, "w") as hf3:
        hf3.create_dataset(
            "MeanVarTest", data=MV, compression="gzip", compression_opts=9
        )

    chemin_fichier_yaml = os.path.join(os.getcwd(), fichier_yaml)

    if os.path.exists(chemin_fichier_yaml):
        shutil.copy(chemin_fichier_yaml, chemin_sous_dossier)
