import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn
import numpy as np
import shutil
import os
import h5py

from gbgpu.gbgpu import GBGPU
from gbgpu.thirdbody import GBGPUThirdBody

from gbgpu.utils.constants import *
from gbgpu.utils.utility import *
from Noise import AnalyticNoise


def generate_catalog(
    f_min,
    f_max,
    num_samples,
    amp_min,
    amp_max,
    iota_min,
    iota_max,
    fdot_min,
    fdot_max,
    phi0_min,
    phi0_max,
    psi_min,
    psi_max,
    lam_min,
    lam_max,
    beta_sky_min,
    beta_sky_max,
):
    f0 = np.random.uniform(f_min, f_max, num_samples)
    amp = np.random.uniform(amp_min, amp_max, num_samples)
    fdot = np.random.uniform(fdot_min, fdot_max, num_samples)
    fddot = np.zeros(num_samples)  # np.random.uniform(1e-50, 1e-49, num_samples)
    phi0 = np.random.uniform(phi0_min, phi0_max, num_samples)
    iota = np.random.uniform(iota_min, iota_max, num_samples)
    psi = np.random.uniform(psi_min, psi_max, num_samples)
    lam = np.random.uniform(lam_min, lam_max, num_samples)
    beta_sky = np.random.uniform(beta_sky_min, beta_sky_max, num_samples)
    return np.array((amp, f0, fdot, fddot, -phi0, iota, psi, lam, beta_sky))


def generate_response(catalog, Tobs, dt, N):
    gb = GBGPU(use_gpu=False)
    gb.run_wave(*catalog, N=N, dt=dt, T=Tobs, oversample=1)
    return gb


def aggregate(a, gb, f_min0, df, start, stop):
    k_min = round(f_min0 / df)
    for i in range(start, stop):
        i_start = gb.start_inds[i - start] - k_min
        i_end = i_start + gb.N
        a[i, i_start:i_end] = gb.A[i - start]


def whiten(a, sample_frequencies):
    noise = AnalyticNoise(sample_frequencies)
    psd_A = noise.psd(option="A")
    asd_A = np.sqrt(psd_A)
    df = sample_frequencies[1] - sample_frequencies[0]
    a *= np.sqrt(4 * df) / asd_A

    ###

    # mean = np.mean(a, axis=0)
    # std = np.std(a, axis=0)
    # b = (a - mean) / std

    # breakpoint()

    ###


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


class ConvMLP(nn.Module):
    def __init__(self, input_channels, sequence_length, hidden_size, output_size):
        super(ConvMLP, self).__init__()

        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)

        hidden_net_list = []
        hidden_net_list.append(nn.Linear(32 * sequence_length, hidden_size[0]))

        for i in range(1, len(hidden_size)):
            hidden_net_list.append(nn.Linear(hidden_size[i - 1], hidden_size[i]))

        self.hidden_net_list = nn.ModuleList(hidden_net_list)
        self.fc = nn.Linear(hidden_size[-1], output_size)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        for layer in self.hidden_net_list:
            x = self.relu(layer(x))
        x = self.fc(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_loop(dataloader, model):
    size = len(dataloader)
    test_loss = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for X, y in dataloader:
            predictions = model(X)
            loss = criterion(predictions, y)
            test_loss += loss.item()
    test_loss /= size
    print(f"Test loss: {test_loss:>8f} \n")
    return test_loss


# def test_loop(dataloader, model):
#     size = len(dataloader)
#     test_loss = 0
#     criterion = nn.MSELoss()
#     with torch.no_grad():
#         for X, y in dataloader:
#             predictions = model(X)
#             loss = criterion(predictions, y)
#             test_loss += loss.item()
#     test_loss /= size
#     print(f"Test loss: {test_loss:>8f} \n")
#     return test_loss


def train_loop(dataloader, model, optimizer):
    size = len(dataloader.dataset)
    train_loss = 0
    criterion = nn.MSELoss()
    for batch, (X, y) in enumerate(dataloader):

        predictions = model(X)

        loss = criterion(predictions, y)

        # train_loss += loss.detach().sum()
        # loss = loss.mean()

        train_loss = train_loss + loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 390 == 0:
            loss_value, current = loss.item(), batch * len(X)
            print(f"Loss: {loss_value:>7f}  [{current:>5d}/{size:>5d} samples]")

    # average_loss = train_loss.item() / size
    avg_loss = train_loss / len(dataloader)
    print(
        "Average loss: {:.4f}".format(avg_loss)
    )  # print("Average loss: {:.4f}".format(avg_loss))
    return avg_loss  # average_loss


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
