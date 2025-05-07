import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import h5py
from Methode import MLP

# === Modifs ===

selected_param_names = ["lam", "beta"]

dossier_train = (
    "1e6_lam_beta_800_800_800_800_e=10_(queblanchiement)_bruittrain_amp=3e-23"
)
dossier_dataset = "1e6_lam_beta_(queblanchiement)_amp=3e-23"
hidden_dims = [800, 800, 800, 800]

# === Liste complète des paramètres disponibles ===
all_param_names = ["amp", "f0", "fdot", "fddot", "phi0", "iota", "psi", "lam", "beta"]

# === Paramètres pour lesquels on veut une erreur absolue (ex: peuvent être négatifs ou angulaires)
absolute_error_names = ["lam", "beta"]

# === Construction des index
selected_indices = [all_param_names.index(p) for p in selected_param_names]
absolute_error_indices = [
    i for i, p in enumerate(selected_param_names) if p in absolute_error_names
]
relative_error_indices = [
    i for i in range(len(selected_param_names)) if i not in absolute_error_indices
]

# === Hyperparamètres modèle ===
input_dim = 378
output_dim = len(selected_param_names)

# === Chemins ===
chemin_base = os.path.join("Train", dossier_train)
chemin_weights = os.path.join(chemin_base, "Weights.pth")
chemin_test = os.path.join(chemin_base, "test_dataset.pth")
chemin_train = os.path.join(chemin_base, "train_dataset.pth")
chemin_mv = os.path.join("Dataset", dossier_dataset, "MeanVar.h5")

# === Chargement modèle et données ===
model = MLP(input_dim, hidden_dims, output_dim)
model.load_state_dict(torch.load(chemin_weights, map_location=torch.device("cpu")))
model.eval()

train_dataset = torch.load(chemin_train, map_location=torch.device("cpu"))
test_dataset = torch.load(chemin_test, map_location=torch.device("cpu"))

with h5py.File(chemin_mv, "r") as hf:
    mean_var = hf["MeanVarTest"][:]


# === Prédictions ===
def predict_dataset(dataset):
    inputs, _ = dataset
    preds = np.zeros((len(inputs), output_dim))
    for i, x in enumerate(inputs):
        y = model(x).detach().numpy()
        preds[i, :] = y
    return preds


param_eval_train = predict_dataset(train_dataset)
param_eval_test = predict_dataset(test_dataset)


# === Extraction robuste des cibles ===
def extract_targets(targets, selected_indices, expected_dim):
    targets_array = np.array(targets)
    if targets_array.shape[1] == expected_dim:
        return targets_array
    else:
        return targets_array[:, selected_indices]


# === Dénormalisation ===
def denormalize(data, mean_var):
    return data * mean_var[1, :] + mean_var[0, :]


true_train = denormalize(
    extract_targets(train_dataset[1], selected_indices, output_dim), mean_var
)
true_test = denormalize(
    extract_targets(test_dataset[1], selected_indices, output_dim), mean_var
)
pred_train = denormalize(param_eval_train, mean_var)
pred_test = denormalize(param_eval_test, mean_var)

# === Conversion en degrés pour lam et beta ===
deg_indices = [
    i for i, name in enumerate(selected_param_names) if name in ["lam", "beta"]
]
radian_to_deg = 180.0 / np.pi

true_train[:, deg_indices] *= radian_to_deg
true_test[:, deg_indices] *= radian_to_deg
pred_train[:, deg_indices] *= radian_to_deg
pred_test[:, deg_indices] *= radian_to_deg

# === Calcul des erreurs absolues ===
# error_abs_train = np.abs(pred_train - true_train)
# error_abs_test = np.abs(pred_test - true_test)

error_abs_train = pred_train - true_train
error_abs_test = pred_test - true_test


# === Correction des erreurs pour paramètres circulaires (ex: lam)
def angular_difference_deg(pred, true):
    delta = pred - true
    return (delta + 180.0) % 360.0 - 180.0


# Paramètres circulaires à corriger
"""
circular_names = ["lam"]  # Ajoute "beta" ici si nécessaire
circular_indices = [
    i for i, name in enumerate(selected_param_names) if name in circular_names
]

for i in circular_indices:
    error_abs_train[:, i] = np.abs(
        angular_difference_deg(pred_train[:, i], true_train[:, i])
    )
    error_abs_test[:, i] = np.abs(
        angular_difference_deg(pred_test[:, i], true_test[:, i])
    )
"""

# === Calcul des erreurs relatives ===
error_rel_train = np.zeros_like(error_abs_train)
error_rel_test = np.zeros_like(error_abs_test)

for idx in relative_error_indices:
    denom_train = np.where(true_train[:, idx] == 0, 1, np.abs(true_train[:, idx]))
    denom_test = np.where(true_test[:, idx] == 0, 1, np.abs(true_test[:, idx]))
    error_rel_train[:, idx] = 100 * error_abs_train[:, idx] / denom_train
    error_rel_test[:, idx] = 100 * error_abs_test[:, idx] / denom_test

# === Moyenne des erreurs ===
mean_errors_train = np.zeros(output_dim)
mean_errors_test = np.zeros(output_dim)

for i in range(output_dim):
    if i in relative_error_indices:
        mean_errors_train[i] = np.mean(np.abs(error_rel_train[:, i]))
        mean_errors_test[i] = np.mean(np.abs(error_rel_test[:, i]))
    else:
        mean_errors_train[i] = np.mean(np.abs(error_abs_train[:, i]))
        mean_errors_test[i] = np.mean(np.abs(error_abs_test[:, i]))

# === Affichage erreurs moyennes ===
print("Erreur moyenne des paramètres sur les données de TRAIN :")
for i, name in enumerate(selected_param_names):
    label = "%" if i in relative_error_indices else "(absolue)"
    print(f" - {name} : {mean_errors_train[i]:.3f} {label}")

print("\nErreur moyenne des paramètres sur les données de TEST :")
for i, name in enumerate(selected_param_names):
    label = "%" if i in relative_error_indices else "(absolue)"
    print(f" - {name} : {mean_errors_test[i]:.3f} {label}")

# === Histogrammes ===
fig, axes = plt.subplots(1, output_dim, figsize=(5 * output_dim, 5))
if output_dim == 1:
    axes = [axes]

for i in range(output_dim):
    errors = (
        error_rel_test[:, i] if i in relative_error_indices else error_abs_test[:, i]
    )
    axes[i].hist(errors, bins=200, edgecolor="black", alpha=0.7)
    ylabel = "Écart relatif (%)" if i in relative_error_indices else "Erreur absolue"
    axes[i].set_xlabel(ylabel)
    axes[i].set_ylabel("Fréquence")
    axes[i].set_title(f"{selected_param_names[i]}")
    axes[i].grid(True)

plt.tight_layout()
plt.show()
