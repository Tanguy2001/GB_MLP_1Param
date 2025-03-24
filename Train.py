import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn
import yaml
import os
import h5py
import numpy as np
import shutil
from Methode import (
    WaveformDataset,
    MLP,
    count_parameters,
    train_loop,
    test_loop,
    enregistrer_donnees,
)


with open("config_Train.yaml", "r") as file:
    config = yaml.safe_load(file)


Dataset_Waveform = config["Dataset_Waveform"]
Dataset_Parametre = config["Dataset_Parametre"]
hidden_dims = config["hidden_dims"]
train_fraction = config["train_fraction"]
batch_size = config["batch_size"]
lr = config["lr"]
epochs = config["epochs"]
alpha = config["alpha"]
Nom_dossier_Dataset = config["Nom_dossier_Dataset"]
nom_dossier_Train = config["nom_dossier_Train"]


chemin_fichier_waveform = os.path.join("Dataset", Nom_dossier_Dataset, Dataset_Waveform)
chemin_fichier_parameter = os.path.join(
    "Dataset", Nom_dossier_Dataset, Dataset_Parametre
)


with h5py.File(chemin_fichier_parameter, "r") as hf:
    parameters_standardized = hf["Parameters_standardized"][:]

parameters_standardized = parameters_standardized.T


with h5py.File(chemin_fichier_waveform, "r") as hf:
    waveforms = hf["Waveform"][:]

waveform_dataset = WaveformDataset(parameters_standardized, waveforms, alpha)

input_dim = waveforms.shape[-1]
output_dim = parameters_standardized.shape[-1]

model = MLP(input_dim, hidden_dims, output_dim)

print(f"Nombre total de paramètres : {count_parameters(model)}")
print(model)

num_samples = waveforms.shape[0]
num_train = int(round(train_fraction * num_samples))
num_test = num_samples - num_train
train_dataset, test_dataset = random_split(waveform_dataset, [num_train, num_test])


train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)


train_features, train_labels = next(iter(train_dataloader))

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


train_history = []
test_history = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    loss = train_loop(train_dataloader, model, optimizer)
    train_history.append(loss)
    loss = test_loop(test_dataloader, model)
    test_history.append(loss)
print("Done!")


dossier_sauvegarde = os.path.join("Train", nom_dossier_Train)
os.makedirs(dossier_sauvegarde, exist_ok=True)

chemin_yaml = os.path.join(os.getcwd(), "config_Train.yaml")
if os.path.exists(chemin_yaml):
    shutil.copy(chemin_yaml, dossier_sauvegarde)

chemin_modele = os.path.join(dossier_sauvegarde, "Weights.pth")
torch.save(model.state_dict(), chemin_modele)


train = train_dataset[0:100]
test = test_dataset[0:100]


chemin_dataloader = os.path.join(dossier_sauvegarde, "test_dataset.pth")
torch.save(test, chemin_dataloader)

chemin_dataloader = os.path.join(dossier_sauvegarde, "train_dataset.pth")
torch.save(train, chemin_dataloader)


epochs = np.arange(1, len(train_history) + 1)
print(type(test_history))
plt.plot(epochs, train_history, label="train loss")
plt.plot(epochs, test_history, label="test loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

chemin_plot = os.path.join(dossier_sauvegarde, "plot.png")
plt.savefig(chemin_plot)

plt.show()
