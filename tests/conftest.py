import numpy as np
import os

num_spins = 10
temperatures = np.linspace(0.1, 3.1, 101)
num_cycles = int(1e4)

filename = os.path.join(
    "tests", "dat", "two_dimensional_{0}.dat".format(num_spins)
)


def exact_energy_one_dimensional(num_spins, temperature):
    return -num_spins * np.tanh(1.0 / temperature)


exact_energies = exact_energy_one_dimensional(num_spins, temperatures)


def read_two_dimensional_energy():
    _dat = np.loadtxt(filename)
    _temperatures = _dat[:, 0]
    _energies = _dat[:, 1]

    assert len(temperatures) == len(_temperatures)

    return _energies


two_dimensional_energies = read_two_dimensional_energy()


def pytest_namespace():
    return {
        "num_spins": num_spins,
        "temperatures": temperatures,
        "num_cycles": num_cycles,
        "exact_energies": exact_energies,
        "two_dimensional_energies": two_dimensional_energies,
    }
