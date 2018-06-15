import numpy as np
import matplotlib.pyplot as plt
import tqdm

from ising_model.one_dimensional import sample_ising


def exact_energy(num_spins, temperature):
    return -num_spins * np.tanh(1.0 / temperature)


temperatures = np.linspace(0.1, 3.0, 101)
num_cycles = int(1e5)

num_spins = 40
sampled_energies = []
exact_energies = []

for temperature in tqdm.tqdm(temperatures):
    exact_energies.append(exact_energy(num_spins, temperature))
    sampled_energies.append(
        sample_ising(
            np.random.choice([-1, 1], num_spins), num_cycles, temperature
        )
    )

plt.plot(temperatures, exact_energies, label="Exact")
plt.plot(temperatures, sampled_energies, label="Sampled")
plt.legend(loc="best")
plt.show()
