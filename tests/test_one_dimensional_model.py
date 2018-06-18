import numpy as np
import pytest

from ising_model.one_dimensional import sample_ising


def test_energy():
    num_spins = pytest.num_spins
    temperatures = pytest.temperatures
    num_cycles = pytest.num_cycles
    exact_energies = pytest.exact_energies

    sampled_energies = np.zeros(len(temperatures))

    for i, temperature in enumerate(temperatures):
        grid = np.ones(num_spins)
        energy = sample_ising(grid, num_cycles, temperature)

        sampled_energies[i] = energy

    sampled_area = np.trapz(sampled_energies, temperatures)
    exact_area = np.trapz(exact_energies, temperatures)

    assert abs(exact_area - sampled_area) < 0.5
