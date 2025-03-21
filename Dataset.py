import numpy as np
import time
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import yaml
import os
from Methode import (
    enregistrer_donnees,
    whiten,
    aggregate,
    generate_response,
    generate_catalog,
)


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
