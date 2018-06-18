import numpy as np
import matplotlib.pyplot as plt
import tqdm

from ising_model.one_dimensional import sample_ising

def exact_energy(num_spins, temperature):
    return -num_spins * np.tanh(1.0 / temperature)


temperatures = np.linspace(0.1, 3.0, 101)
num_cycles = int(1e4)

num_spins = 40
sampled_energies = np.zeros_like(temperatures)

for i, temperature in enumerate(tqdm.tqdm(temperatures)):
    grid = np.ones(num_spins)
    energy = sample_ising(grid, num_cycles, temperature)

    sampled_energies[i] = energy

exact_energies = exact_energy(num_spins, temperatures)
plt.plot(temperatures, exact_energies, label="Exact")
plt.plot(temperatures, sampled_energies, label="Sampled")
plt.legend(loc="best")
plt.show()
