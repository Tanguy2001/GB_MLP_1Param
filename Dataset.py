import numpy as np
import time
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import yaml
import os
from Methode import enregistrer_donnees


from gbgpu.gbgpu import GBGPU
from gbgpu.thirdbody import GBGPUThirdBody

from gbgpu.utils.constants import *
from gbgpu.utils.utility import *
from Noise import AnalyticNoise


with open("config_Dataset.yaml", "r") as file:
    config = yaml.safe_load(file)


f_min = config["f_min"]
f_max = f_min + float(config["interval_freq"])
num_samples = int(float(config["num_samples"]))
n = config["Tobs"]
dt = config["dt"]
N = config["N"]
chunk_size = config["chunk_size"]
Nom_Waveforms_h5 = config["Nom_Waveforms_h5"]
Nom_Parametres_h5 = config["Nom_Parametres_h5"]
Nom_MeanVar_h5 = config["Nom_MeanVar_h5"]
nom_dossier_Dataset = config["nom_dossier_Dataset"]


def generate_catalog(f_min, f_max, num_samples):
    f0 = np.random.uniform(f_min, f_max, num_samples)
    amp = np.random.uniform(1e-23, 1e-21, num_samples)
    fdot = np.random.uniform(7.538331e-16, 7.538331e-18, num_samples)
    fddot = np.zeros(num_samples)  # np.random.uniform(1e-50, 1e-49, num_samples)
    phi0 = np.random.uniform(0, 2 * np.pi, num_samples)
    iota = np.random.uniform(0, np.pi, num_samples)
    psi = np.random.uniform(0, np.pi, num_samples)
    lam = np.random.uniform(-np.pi, np.pi, num_samples)
    beta_sky = np.random.uniform(-np.pi / 2, np.pi / 2, num_samples)
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


def generate(n, f_min, f_max, num_samples, chunk_size, N, dt):

    Tobs = n * YEAR
    df = 1.0 / Tobs
    f_min0 = f_min - 3 * 1e-6
    f_max0 = f_max + 3 * 1e-6
    sample_frequencies = np.arange(f_min0, f_max0, df)
    A_whitened = np.zeros((num_samples, len(sample_frequencies)), dtype=np.complex64)

    cat = generate_catalog(f_min, f_max, num_samples)

    for start in tqdm(
        range(0, num_samples, chunk_size), desc="Processing", unit="chunk"
    ):
        stop = min(start + chunk_size, cat.shape[1])
        gb = generate_response(cat[:, start:stop], Tobs, dt, N)
        aggregate(A_whitened, gb, f_min0, df, start, stop)
        del gb
    whiten(A_whitened, sample_frequencies)

    waveforms = np.hstack((A_whitened.real, A_whitened.imag))

    parameters_mean = np.mean(cat, axis=1)
    parameters_std = np.std(cat, axis=1)
    parameters_standardized = ((cat.T - parameters_mean) / parameters_std).T
    parameters_standardized = np.nan_to_num(parameters_standardized, nan=0)

    MV = [parameters_mean, parameters_std]

    enregistrer_donnees(
        "Dataset",
        Nom_Waveforms_h5,
        waveforms,
        "config_Dataset.yaml",
        Nom_Parametres_h5,
        parameters_standardized,
        Nom_MeanVar_h5,
        MV,
        nom_dossier_Dataset,
    )

    print("LE CODE TOURNE SANS ERREUR")


if __name__ == "__main__":
    generate(n, f_min, f_max, num_samples, chunk_size, N, dt)
