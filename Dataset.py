import numpy as np
import time
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
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


F_max = config["F_max"]
f_min = config["f_min"]

if F_max == 1:
    f_max = f_min + float(config["interval_freq"])
else:
    f_max = f_min

num_samples = int(float(config["num_samples"]))
n = config["Tobs"]
dt = config["dt"]
N = config["N"]
chunk_size = config["chunk_size"]
Nom_Waveforms_h5 = config["Nom_Waveforms_h5"]
Nom_Parametres_h5 = config["Nom_Parametres_h5"]
Nom_MeanVar_h5 = config["Nom_MeanVar_h5"]
nom_dossier_Dataset = config["nom_dossier_Dataset"]
amp_min = float(config["amp_min"])
amp_max = float(config["amp_max"])
iota_min = config["iota_min"]
iota_max = config["iota_max"]
fdot_min = config["fdot_min"]
fdot_max = config["fdot_max"]
phi0_min = config["phi0_min"]
phi0_max = config["phi0_max"]
psi_min = config["psi_min"]
psi_max = config["psi_max"]
lam_min = config["lam_min"]
lam_max = config["lam_max"]
beta_sky_min = config["beta_sky_min"]
beta_sky_max = config["beta_sky_max"]


def generate(
    n,
    f_min,
    f_max,
    num_samples,
    chunk_size,
    N,
    dt,
    amp_min,
    amp_max,
    iota_min,
    iota_max,
):

    Tobs = n * YEAR
    df = 1.0 / Tobs
    f_min0 = f_min - 3 * 1e-6
    f_max0 = f_max + 3 * 1e-6
    sample_frequencies = np.arange(f_min0, f_max0, df)
    A_whitened = np.zeros((num_samples, len(sample_frequencies)), dtype=np.complex64)

    cat = generate_catalog(
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
    )

    for start in tqdm(
        range(0, num_samples, chunk_size), desc="Processing", unit="chunk"
    ):
        stop = min(start + chunk_size, cat.shape[1])
        gb = generate_response(cat[:, start:stop], Tobs, dt, N)
        aggregate(A_whitened, gb, f_min0, df, start, stop)
        del gb

    whiten(A_whitened, sample_frequencies)

    waveforms = np.hstack((A_whitened.real, A_whitened.imag))

    scaler = MinMaxScaler()
    waveforms = scaler.fit_transform(waveforms)

    # fig, axes = plt.subplots(1, 2, figsize=(10, 10))
    # axes[0].plot(waveforms[0])
    # axes[1].plot(waveforms1[0])
    # plt.show()

    # for i in range(len(cat)):
    #     if cat[i][0] == cat[i][1]:
    #         cat = np.delete(cat, i, axis=0)

    mask = np.array(
        [True if cat[i][0] != cat[i][1] else False for i in range(len(cat))]
    )
    cat = cat[mask]

    parameters_mean = np.mean(cat, axis=1)
    parameters_std = np.std(cat, axis=1)
    parameters_standardized = ((cat.T - parameters_mean) / parameters_std).T
    # parameters_standardized = np.nan_to_num(parameters_standardized, nan=0)

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
    generate(
        n,
        f_min,
        f_max,
        num_samples,
        chunk_size,
        N,
        dt,
        amp_min,
        amp_max,
        iota_min,
        iota_max,
    )
