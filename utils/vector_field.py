import numpy as np
import matplotlib.pyplot as plt

# Define the SELU activation function and its gradient
def selu(x, alpha=1.67326, scale=1.0507):
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))

def grad_selu(x, alpha=1.67326, scale=1.0507):
    return scale * np.where(x > 0, 1, alpha * np.exp(x))

# Define a function to compute the next layer's mean and variance
def next_layer_stats(mu, nu, alpha=1.67326, scale=1.0507):
    mu_tilde = scale * (mu * np.where(mu > 0, 1, alpha * np.exp(mu)) +
                        (1 - mu) * alpha * np.exp(mu) * np.where(mu <= 0, 1, 0))
    nu_tilde = scale**2 * (nu * np.where(mu > 0, 1, alpha * np.exp(mu))**2 +
                           (1 - mu) * alpha**2 * np.exp(2 * mu) * np.where(mu <= 0, 1, 0))
    return mu_tilde, nu_tilde

# Generate a grid of points in the (mu, nu) space
mu_values = np.linspace(-0.1, 0.1, 20)
nu_values = np.linspace(0.8, 1.5, 20)
MU, NU = np.meshgrid(mu_values, nu_values)

# Compute the gradients
MU_TILDE = np.zeros_like(MU)
NU_TILDE = np.zeros_like(NU)

for i in range(MU.shape[0]):
    for j in range(MU.shape[1]):
        mu_tilde, nu_tilde = next_layer_stats(MU[i, j], NU[i, j])
        MU_TILDE[i, j] = mu_tilde - MU[i, j]
        NU_TILDE[i, j] = nu_tilde - NU[i, j]

# Plot the vector field
plt.figure(figsize=(10, 5))
plt.quiver(NU, MU, NU_TILDE, MU_TILDE, angles='xy')
plt.xlabel(r'$\nu / \tilde{\nu}$')
plt.ylabel(r'$\mu / \tilde{\mu}$')
plt.title('Gradients Pointing to a Local Minimum Using SELU Activation Function')
plt.grid()
plt.show()