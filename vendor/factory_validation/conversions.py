import numpy as np


def sound_speed_from_temperature_c(temperature_c):
    temperature_c = np.asarray(temperature_c, dtype=np.float64)
    return 331.3 * np.sqrt(1.0 + temperature_c / 273.15)


def temperature_from_sound_speed(sound_speed):
    sound_speed = np.asarray(sound_speed, dtype=np.float64)
    return 273.15 * ((sound_speed / 331.3) ** 2 - 1.0)


def path_temperature_to_tof(path_temperature_c, path_length_m):
    path_temperature_c = np.asarray(path_temperature_c, dtype=np.float64)
    path_length_m = np.asarray(path_length_m, dtype=np.float64)
    return path_length_m / sound_speed_from_temperature_c(path_temperature_c)


def tof_to_path_temperature(tof_s, path_length_m):
    tof_s = np.asarray(tof_s, dtype=np.float64)
    path_length_m = np.asarray(path_length_m, dtype=np.float64)
    tof_safe = np.where(np.abs(tof_s) < 1e-14, 1e-14, tof_s)
    return temperature_from_sound_speed(path_length_m / tof_safe)

