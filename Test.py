import torch
import numpy as np
from Methode import MLP
import os


input_dim = 756
hidden_dims = [1000, 1000]
output_dim = 9

dosier_train = "Entrainement_1e6"


model = MLP(input_dim, hidden_dims, output_dim)

chemin_train = os.path.join("Train", dosier_train)
chemin_weights = os.path.join(chemin_train, "Weights.pth")
chemin_dataloader = os.path.join(chemin_train, "train_dataloader.pth")


model.load_state_dict(torch.load(chemin_weights, map_location=torch.device("cpu")))
test_dataloader = torch.load(
    chemin_dataloader, map_location=torch.device("cpu"), weights_only=False
)


model.eval()

waveform = []
param = []

for batch in test_dataloader:
    w, p = batch
    waveform = waveform + w.tolist()
    param = param + p.tolist()

w_tensor = torch.tensor(waveform).float()


outputs = model(w_tensor).tolist()

outputs = np.array(outputs)
param = np.array(param)
param_safe = np.where(param == 0, np.nan, param)

ecart_relatif = 100 * np.abs(outputs - param) / np.abs(param)
ecart_relatif = np.mean(ecart_relatif, axis=0)
# ecart_relatif = np.floor(ecart_relatif).astype(int)


breakpoint()
