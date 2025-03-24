import torch
import numpy as np
from Methode import MLP
import os


input_dim = 756
hidden_dims = [1000, 1000]
output_dim = 9

dosier_train = "Entrainement_1e5"


model = MLP(input_dim, hidden_dims, output_dim)

chemin = os.path.join("Train", dosier_train)
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

relative_error_train = (
    np.abs(parametre_train - parametre_eval_train) / denominateur_train
)
relative_error_test = np.abs(parametre_test - parametre_eval_test) / denominateur_test

np.set_printoptions(precision=6, suppress=True)

moyenne_train = np.mean(relative_error_train, axis=0)

moyenne_test = np.mean(relative_error_test, axis=0)


print("Ecart relatif moyen des paramètres sur les données de train :", moyenne_train)

print("Ecart relatif moyen des paramètres sur les données de test :", moyenne_test)
