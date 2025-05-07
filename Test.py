import torch
import numpy as np
from Methode import MLP
import os
import matplotlib.pyplot as plt
import h5py

input_dim = 378
hidden_dims = [800, 800, 800, 800]
output_dim = 2

dossier_train = "1e6_lam_beta_800_800_800_800_e=20"
dossier_dataset = "1e6_lam_beta"

model = MLP(input_dim, hidden_dims, output_dim)

chemin = os.path.join("Train", dossier_train)
chemin_weights = os.path.join(chemin, "Weights.pth")
chemin_test = os.path.join(chemin, "test_dataset.pth")
chemin_train = os.path.join(chemin, "train_dataset.pth")

Chemin = os.path.join("Dataset", dossier_dataset)
chemin_MV = os.path.join(Chemin, "MeanVar.h5")

with h5py.File(chemin_MV, "r") as hf:
    MeanVar = hf["MeanVarTest"][:]


model.load_state_dict(torch.load(chemin_weights, map_location=torch.device("cpu")))
test_dataset = torch.load(
    chemin_test, map_location=torch.device("cpu"), weights_only=False
)

train_dataset = torch.load(
    chemin_train, map_location=torch.device("cpu"), weights_only=False
)


model.eval()

waveform_test = test_dataset[0][0]
param_test = test_dataset[1][0]
param_eval_test = model(waveform_test)


waveform_train = train_dataset[0][0]
param_train = train_dataset[1][0]
param_eval_train = model(waveform_train)


parametre_eval_train = np.zeros((len(train_dataset[0]), len(train_dataset[1][0])))
parametre_eval_test = np.zeros((len(train_dataset[0]), len(train_dataset[1][0])))


for i in range(len(train_dataset[0])):
    w_test = test_dataset[0][i]
    p_test = model(w_test).detach().numpy()
    parametre_eval_test[i][:] = p_test

    w_train = train_dataset[0][i]
    p_train = model(w_train).detach().numpy()
    parametre_eval_train[i][:] = p_train


parametre_train = np.array(train_dataset[1][:])
parametre_test = np.array(test_dataset[1][:])


################### Erreur absolue dénormalisee ####################

parametre_train = parametre_train * MeanVar[1, :] + MeanVar[0, :]
parametre_test = parametre_test * MeanVar[1, :] + MeanVar[0, :]

parametre_eval_train = parametre_eval_train * MeanVar[1, :] + MeanVar[0, :]
parametre_eval_test = parametre_eval_test * MeanVar[1, :] + MeanVar[0, :]

AMP_mean_test = np.mean(
    np.abs(
        100
        * (parametre_test[:, 0:2] - parametre_eval_test[:, 0:2])
        / parametre_test[:, 0:2]
    )
)

AMP_mean_train = np.mean(
    np.abs(
        100
        * (parametre_train[:, 0:2] - parametre_eval_train[:, 0:2])
        / parametre_train[:, 0:2]
    )
)


relative_error_train = np.abs(parametre_train - parametre_eval_train)
relative_error_test = np.abs(parametre_test - parametre_eval_test)


"""

################### Erreur relative ####################

denominateur_train = np.where(parametre_train == 0, 1, np.abs(parametre_train))
denominateur_test = np.where(parametre_test == 0, 1, np.abs(parametre_test))


relative_error_train = 100 * (
    np.abs(parametre_train - parametre_eval_train) / denominateur_train
)
relative_error_test = (
    100 * np.abs(parametre_test - parametre_eval_test) / denominateur_test
)

"""


np.set_printoptions(precision=6, suppress=True)

moyenne_train = np.mean(relative_error_train, axis=0)

moyenne_test = np.mean(relative_error_test, axis=0)

parametres = ["amp", "f0", "fdot", "fddot", "-phi0", "iota", "psi", "lam", "beta_sky"]
parametres = ["lam", "beta_sky"]


fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()

for i in range(len(relative_error_test[0])):
    axes[i].hist(relative_error_test[:, i], bins=10000, edgecolor="black", alpha=0.7)
    axes[i].set_xlabel("Écart relatif (%)")
    axes[i].set_ylabel("Fréquence")
    axes[i].set_title(f"Histogramme de {parametres[i]}")
    axes[i].grid(True)
    # axes[i].set_xlim(-10, 100)

for j in range(len(relative_error_test[0]), len(axes)):
    fig.delaxes(axes[j])

plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()


moyenne_train[0:2] = AMP_mean_train

moyenne_test[0:2] = AMP_mean_test


print("Ecart relatif moyen des paramètres sur les données de train :", moyenne_train)

print("Ecart relatif moyen des paramètres sur les données de test :", moyenne_test)
