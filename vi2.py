import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import uniform, normal, laplace

# Učitavanje podataka
data = pd.read_csv('observations.csv', header=None)
rho_measurements = data[0].values
theta_measurements = data[1].values

# Parametri
num_particles = 1000
num_timesteps = len(rho_measurements)
velocity = 0.5  # m/s
sigma_rho_tangent = 0.3
sigma_rho_radial = 0.6
sigma_theta = np.pi / 36

# Inicijalizacija čestica
particles = np.empty((num_particles, 4))  # x, y, theta, weight
particles[:, 0] = uniform(-2, 2, size=num_particles)  # x
particles[:, 1] = uniform(-2, 2, size=num_particles)  # y
particles[:, 2] = uniform(-np.pi, np.pi, size=num_particles)  # theta
particles[:, 3] = 1.0 / num_particles  # weight

# Funkcije za ažuriranje
def motion_model(particles):
    particles[:, 0] += velocity * np.cos(particles[:, 2])  # x
    particles[:, 1] += velocity * np.sin(particles[:, 2])  # y

def measurement_model(particles, rho, theta):
    x_diff = particles[:, 0]
    y_diff = particles[:, 1]
    distance = np.sqrt(x_diff**2 + y_diff**2)
    bearing = np.arctan2(y_diff, x_diff)
    
    sigma_rho = np.where(np.abs(np.sin(particles[:, 2] - bearing)) > 0.5, sigma_rho_radial, sigma_rho_tangent)
    
    weight_rho = np.exp(-0.5 * ((distance - rho) / sigma_rho) ** 2) / (sigma_rho * np.sqrt(2 * np.pi))
    weight_theta = np.exp(-np.abs(theta - bearing) / sigma_theta) / (2 * sigma_theta)
    
    particles[:, 3] = weight_rho * weight_theta
    particles[:, 3] += 1.e-300  # Avoid zeros
    particles[:, 3] /= np.sum(particles[:, 3])  # Normalize

def resample(particles):
    indices = np.random.choice(range(num_particles), size=num_particles, p=particles[:, 3])
    particles = particles[indices]
    particles[:, 3] = 1.0 / num_particles
    return particles

# Čestični filter
for t in range(num_timesteps):
    motion_model(particles)
    measurement_model(particles, rho_measurements[t], theta_measurements[t])
    
    if t % 5 == 0:  # Resampling condition
        particles = resample(particles)
    
    # Crtanje za prva dva koraka
    if t < 2:
        plt.figure(figsize=(10, 5))
        plt.scatter(particles[:, 0], particles[:, 1], s=particles[:, 3] * 10000, alpha=0.6)
        plt.scatter(rho_measurements[t] * np.cos(theta_measurements[t]), rho_measurements[t] * np.sin(theta_measurements[t]), c='r', marker='x')
        plt.title(f"Step {t + 1}")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

# Prikazivanje procenjenih i stvarnih putanja
estimated_positions = np.average(particles[:, :2], weights=particles[:, 3], axis=0)
estimated_path = [estimated_positions]

for t in range(num_timesteps):
    motion_model(particles)
    measurement_model(particles, rho_measurements[t], theta_measurements[t])
    
    if t % 5 == 0:
        particles = resample(particles)
    
    estimated_positions = np.average(particles[:, :2], weights=particles[:, 3], axis=0)
    estimated_path.append(estimated_positions)

estimated_path = np.array(estimated_path)

plt.figure(figsize=(10, 5))
plt.plot(estimated_path[:, 0], estimated_path[:, 1], label='Estimated Path')
plt.scatter(rho_measurements * np.cos(theta_measurements), rho_measurements * np.sin(theta_measurements), c='r', marker='x', label='Measurements')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Estimated and Measured Path')
plt.show()