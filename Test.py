import torch
import numpy as np
from Methode import MLP
import os
import matplotlib.pyplot as plt

input_dim = 378
hidden_dims = [300, 300]
output_dim = 3

dossier_train = "1e6_amp_f0_iota_phi0_fdot_fixe_300_300_e=20"


model = MLP(input_dim, hidden_dims, output_dim)

chemin = os.path.join("Train", dossier_train)
chemin_weights = os.path.join(chemin, "Weights.pth")
chemin_test = os.path.join(chemin, "test_dataset.pth")
chemin_train = os.path.join(chemin, "train_dataset.pth")

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


denominateur_train = np.where(parametre_train == 0, 1, np.abs(parametre_train))
denominateur_test = np.where(parametre_test == 0, 1, np.abs(parametre_test))

relative_error_train = 100 * (
    np.abs(parametre_train - parametre_eval_train) / denominateur_train
)
relative_error_test = (
    100 * np.abs(parametre_test - parametre_eval_test) / denominateur_test
)

np.set_printoptions(precision=6, suppress=True)

moyenne_train = np.mean(relative_error_train, axis=0)

moyenne_test = np.mean(relative_error_test, axis=0)

parametres = ["amp", "f0", "fdot", "fddot", "-phi0", "iota", "psi", "lam", "beta_sky"]
parametres = ["psi", "lam", "beta_sky"]


fig, axes = plt.subplots(3, 3, figsize=(10, 10))
axes = axes.flatten()

for i in range(len(relative_error_test[0])):
    axes[i].hist(relative_error_test[:, i], bins=100, edgecolor="black", alpha=0.7)
    axes[i].set_xlabel("Écart relatif (%)")
    axes[i].set_ylabel("Fréquence")
    axes[i].set_title(f"Histogramme de {parametres[i]}")
    axes[i].grid(True)
    axes[i].set_xlim(-10, 200)

for j in range(len(relative_error_test[0]), len(axes)):
    fig.delaxes(axes[j])

plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()

print("Ecart relatif moyen des paramètres sur les données de train :", moyenne_train)

print("Ecart relatif moyen des paramètres sur les données de test :", moyenne_test)
