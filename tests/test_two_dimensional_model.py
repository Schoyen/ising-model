import numpy as np
import pytest

from ising_model.two_dimensional import sample_ising


def test_energy():
    num_spins = pytest.num_spins
    temperatures = pytest.temperatures
    num_cycles = pytest.num_cycles
    two_dimensional_energies = pytest.two_dimensional_energies

    sampled_energies = np.zeros(len(temperatures))

    for i, temperature in enumerate(temperatures):
        grid = np.ones((num_spins, num_spins))
        energy = sample_ising(grid, num_cycles, temperature)

        sampled_energies[i] = energy

    sampled_energies *= 1.0 / (num_spins ** 2)
    sampled_area = np.trapz(sampled_energies, temperatures)
    two_dimensional_area = np.trapz(two_dimensional_energies, temperatures)

    assert abs(sampled_area - two_dimensional_area) < 0.5
