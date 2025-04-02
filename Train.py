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
    ConvMLP,
)


with open("config_Train.yaml", "r") as file:
    config = yaml.safe_load(file)


Dataset_Waveform = config["Dataset_Waveform"]
Dataset_Parametre = config["Dataset_Parametre"]
Fichier_MeanVar = config["MeanVar"]
hidden_dims = config["hidden_dims"]
train_fraction = config["train_fraction"]
batch_size = config["batch_size"]
lr = config["lr"]
epochs = config["epochs"]
alpha = config["alpha"]
Nom_dossier_Dataset = config["Nom_dossier_Dataset"]
nom_dossier_Train = config["nom_dossier_Train"]
taille_histo = config["taille_histo"]


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

parametres = ["amp", "f0", "fdot", "fddot", "-phi0", "iota", "psi", "lam", "beta_sky"]
parametres = ["psi", "lam", "beta_sky"]


if len(parametres) != output_dim:
    raise ValueError(
        f"Erreur : la taille des paramètres ({len(parametres)}) ne correspond pas à input_dim ({output_dim})"
    )


dossier_sauvegarde = os.path.join("Train", nom_dossier_Train)

os.makedirs(dossier_sauvegarde, exist_ok=True)

chemin_yaml = os.path.join(os.getcwd(), "config_Train.yaml")
if os.path.exists(chemin_yaml):
    shutil.copy(chemin_yaml, dossier_sauvegarde)


model = MLP(input_dim, hidden_dims, output_dim)
# model = ConvMLP(1, input_dim, hidden_dims, output_dim)  #####################

print(f"Nombre total de paramètres : {count_parameters(model)}")
print(model)

num_samples = waveforms.shape[0]
num_train = int(round(train_fraction * num_samples))
num_test = num_samples - num_train
train_dataset, test_dataset = random_split(waveform_dataset, [num_train, num_test])


train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)


train_features, train_labels = next(iter(train_dataloader))

optimizer = torch.optim.Adam(model.parameters(), lr)


chemin_fichier_MeanVar = os.path.join("Dataset", Nom_dossier_Dataset, Fichier_MeanVar)

with h5py.File(chemin_fichier_MeanVar, "r") as hf:
    MeanVar = hf["MeanVarTest"][:]


train_history = []
test_history = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    loss = train_loop(train_dataloader, model, optimizer)
    train_history.append(loss)
    loss = test_loop(test_dataloader, model)
    test_history.append(loss)
print("Done!")


chemin_modele = os.path.join(dossier_sauvegarde, "Weights.pth")
torch.save(model.state_dict(), chemin_modele)


train = train_dataset[0:taille_histo]
test = test_dataset[0:taille_histo]


chemin_dataloader = os.path.join(dossier_sauvegarde, "test_dataset.pth")
torch.save(test, chemin_dataloader)

chemin_dataloader = os.path.join(dossier_sauvegarde, "train_dataset.pth")
torch.save(train, chemin_dataloader)


epochs = np.arange(1, len(train_history) + 1)
print(type(test_history))
plt.plot(epochs, train_history, label="train loss")
plt.plot(epochs, test_history, label="val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

chemin_plot = os.path.join(dossier_sauvegarde, "plot.png")
plt.savefig(chemin_plot)

plt.show()


# Histogramme des ecarts relatifs


model = MLP(input_dim, hidden_dims, output_dim)

chemin = os.path.join("Train", nom_dossier_Train, "Weights.pth")
model.load_state_dict(torch.load(chemin, map_location=torch.device("cpu")))

model.eval()

parametre_eval_train = np.zeros((len(train[0]), len(train[1][0])))
parametre_eval_test = np.zeros((len(train[0]), len(train[1][0])))


for i in range(len(train[0])):
    w_test = test[0][i]
    p_test = model(w_test).detach().numpy()
    parametre_eval_test[i][:] = p_test

    w_train = train[0][i]
    p_train = model(w_train).detach().numpy()
    parametre_eval_train[i][:] = p_train


parametre_train = np.array(train[1][:])
parametre_test = np.array(test[1][:])

denominateur_train = np.where(parametre_train == 0, 1, np.abs(parametre_train))
denominateur_test = np.where(parametre_test == 0, 1, np.abs(parametre_test))

relative_error_train = (
    np.abs(parametre_train - parametre_eval_train) / denominateur_train
)
relative_error_test = (
    100 * np.abs(parametre_test - parametre_eval_test) / denominateur_test
)


fig, axes = plt.subplots(3, 3, figsize=(10, 10))
axes = axes.flatten()

for i in range(len(relative_error_test[0])):
    axes[i].hist(relative_error_test[:, i], bins=100, edgecolor="black", alpha=0.7)
    axes[i].set_xlabel("Écart relatif (%)")
    axes[i].set_ylabel("Fréquence")
    axes[i].set_title(f"Histogramme de {parametres[i]}")
    axes[i].grid(True)

for j in range(len(relative_error_test[0]), len(axes)):
    fig.delaxes(axes[j])

plt.subplots_adjust(wspace=0.5, hspace=0.5)

chemin_plot = os.path.join(dossier_sauvegarde, "Histogrammes.png")
plt.savefig(chemin_plot)
plt.show()
