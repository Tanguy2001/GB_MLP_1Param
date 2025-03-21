import numpy as np


def add_Signals(Signals, Freqs, N):
    """
    Signals : Matrice NxP, P signaux et N pts par signal
    Freqs : Matrice NxP, P listes de frequence de taille N

    """

    f1 = np.min(Freqs[:, 0])
    f2 = np.max(Freqs[:, 0])
    df = (f2 - f1) / (N - 1)

    f_min = np.min(Freqs[0])
    f_max = np.max(Freqs[-1])

    Signals_new = []
    Freqs_new = []

    for i in range(Signals.shape[1]):
        signal = Signals[:, i]
        freqs = Freqs[:, i]

        Np_min = round(abs(np.min(freqs) - f_min) / df)
        Np_max = round(abs(np.max(freqs) - f_max) / df)

        if Np_min == 0:
            index_f_fin = Np_max

        signal = np.concatenate(
            (
                np.repeat(signal[0], Np_min),
                np.transpose(signal),
                np.repeat(signal[-1], Np_max),
            )
        )
        Signals_new.append(signal)

    Signals_new = np.transpose(Signals_new)
    min_index = np.unravel_index(np.argmin(Freqs), Freqs.shape)
    max_index = np.unravel_index(np.argmax(Freqs), Freqs.shape)

    fa = Freqs[:, min_index[1]]
    fb = Freqs[:, max_index[1]]

    f = np.concatenate((fa, fb[-index_f_fin:]))

    S = np.sum(Signals_new, axis=1)

    return S, f


def adjust_frequency_range(signal, frequencies, target_range):
    """
    Ajuste la plage de fréquences d'un signal pour qu'elle soit exactement target_range Hz.

    - Coupe les extrémités si la plage est trop large.
    - Ajoute des points aux extrémités si la plage est trop courte.

    Arguments :
    - signal : ndarray (N,) -> Les valeurs du signal
    - frequencies : ndarray (N,) -> Les fréquences associées
    - target_range : float -> La plage de fréquences souhaitée (par défaut 10 µHz)

    Retourne :
    - signal_adjusted : ndarray -> Signal ajusté
    - frequencies_adjusted : ndarray -> Fréquences ajustées
    """

    f_min, f_max = frequencies[0], frequencies[-1]
    current_range = f_max - f_min

    if current_range > target_range:
        # Trop large → On coupe de chaque côté
        excess = current_range - target_range
        cut = excess / 2

        # Trouver les indices où couper
        f_start = f_min + cut
        f_end = f_max - cut
        mask = (frequencies >= f_start) & (frequencies <= f_end)

        return signal[mask], frequencies[mask]

    elif current_range < target_range:
        # Trop court → On ajoute des points
        missing = target_range - current_range
        extra_points = int(np.ceil(missing / np.mean(np.diff(frequencies))))

        # Ajouter des points au début
        left_padding_freqs = np.linspace(
            f_min - missing / 2, f_min, extra_points // 2, endpoint=False
        )
        left_padding_signal = np.full_like(left_padding_freqs, signal[0], dtype=complex)

        # Ajouter des points à la fin
        right_padding_freqs = np.linspace(
            f_max, f_max + missing / 2, extra_points // 2, endpoint=True
        )
        right_padding_signal = np.full_like(
            right_padding_freqs, signal[-1], dtype=complex
        )

        # Combiner le tout
        frequencies_adjusted = np.concatenate(
            (left_padding_freqs, frequencies, right_padding_freqs)
        )
        signal_adjusted = np.concatenate(
            (left_padding_signal, signal, right_padding_signal)
        )

        return signal_adjusted, frequencies_adjusted

    else:
        # Déjà correct
        return signal, frequencies


def Somme_2signaux_chevauche(Signal1, Signal2, Freq1, Freq2):
    """
    Signals : Matrice NxP, P signaux et N pts par signal
    Freqs : Matrice NxP, P listes de frequence de taille N

    """

    f1 = np.min(Freq1)
    f2 = np.max(Freq1)
    df = (f2 - f1) / (len(Freq1) - 1)

    fmin = np.min((np.min(Freq1), np.min(Freq2)))
    fmax = np.max((np.max(Freq1), np.max(Freq2)))

    Np_min1 = round(abs(np.min(Freq1) - fmin) / df)
    Np_max1 = round(abs(np.max(Freq1) - fmax) / df)

    Np_min2 = round(abs(np.min(Freq2) - fmin) / df)
    Np_max2 = round(abs(np.max(Freq2) - fmax) / df)

    signal1 = np.concatenate(
        (np.repeat(Signal1[0], Np_min1), Signal1, np.repeat(Signal1[-1], Np_max1))
    )
    signal2 = np.concatenate(
        (np.repeat(Signal2[0], Np_min2), Signal2, np.repeat(Signal2[-1], Np_max2))
    )

    S = signal1 + signal2
    f = np.sort(np.unique(np.concatenate((Freq1, Freq2))))

    return S, f


def are_disjoint(freqs1, freqs2):
    """
    Vérifie si les plages de fréquences de deux signaux sont disjointes.
    """
    f_min1, f_max1 = freqs1[0], freqs1[-1]
    f_min2, f_max2 = freqs2[0], freqs2[-1]

    # Les intervalles sont disjoints si l'un est avant l'autre
    return f_max1 < f_min2 or f_max2 < f_min1


def Somme_2signaux_disjoint(signal1, freqs1, signal2, freqs2):
    """
    Crée un vecteur de fréquence combiné et un signal qui colle les deux signaux,
    en remplissant l'espace entre les plages disjointes.
    """

    # Plages disjointes

    if np.min(freqs1) < np.min(freqs2):
        p, m = freqs1[-1], freqs2[0]
        N = int((m - p) / np.mean(np.diff(freqs1)))
        f_combined = np.concatenate([freqs1, np.linspace(p, m, N), freqs2])
        signal_combined = np.concatenate([signal1, np.repeat(signal1[-1], N), signal2])

    else:
        p, m = freqs2[-1], freqs1[0]
        N = int((m - p) / np.mean(np.diff(freqs1)))
        f_combined = np.concatenate([freqs2, np.linspace(p, m, N), freqs1])
        signal_combined = np.concatenate([signal2, np.repeat(signal2[-1], N), signal1])

    return signal_combined, f_combined


def add_Signals2(Signals, Freqs):

    s = Signals[:, 0]
    f = Freqs[:, 0]

    for i in range(1, Signals.shape[1]):

        if are_disjoint(f, Freqs[:, i]):
            s, f = Somme_2signaux_disjoint(s, f, Signals[:, i], Freqs[:, i])
        else:
            s, f = Somme_2signaux_chevauche(s, Signals[:, i], f, Freqs[:, i])
    return s, f
