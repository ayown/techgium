import numpy as np


def generate_hr_spo2_sequence(length=1024, anomaly=False):
    t = np.linspace(0, 10, length)

    hr = 70 + 5 * np.sin(2 * np.pi * 0.2 * t)
    spo2 = 97 + 0.5 * np.sin(2 * np.pi * 0.1 * t)

    if anomaly:
        hr += np.random.uniform(15, 30)
        spo2 -= np.random.uniform(3, 8)

    hr += np.random.normal(0, 1, length)
    spo2 += np.random.normal(0, 0.2, length)

    return np.stack([hr, spo2], axis=1)
    