import numpy as np
import matplotlib.pyplot as plt
import tqdm

from ising_model.two_dimensional import sample_ising

temperatures = np.linspace(0.1, 3.0, 101)
num_cycles = int(1e4)

num_spins = 40
sampled_energies = np.zeros_like(temperatures)

for i, temperature in enumerate(tqdm.tqdm(temperatures)):
    grid = np.ones((num_spins, num_spins))
    energy = sample_ising(grid, num_cycles, temperature)

    sampled_energies[i] = energy

plt.plot(temperatures, sampled_energies, label="Sampled")
plt.legend(loc="best")
plt.show()
