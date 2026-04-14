"""
TP RIC - Semaine 2 : Architecture du MLP + Forward Pass
Université de Jijel - M1 Intelligence Artificielle

Architecture : [2, 64, 64, 1]
"""

import numpy as np

# ─────────────────────────────────────────────
# Fonctions d'activation
# ─────────────────────────────────────────────
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


# ─────────────────────────────────────────────
# Classe MLP
# ─────────────────────────────────────────────
class MLP:
    def __init__(self, layer_sizes):
        """
        layer_sizes : liste des tailles de chaque couche
        ex: [2, 64, 64, 1]
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.weights = []
        self.biases = []
        self._init_params()

    def _init_params(self):
        """Initialisation He pour les couches ReLU"""
        for i in range(self.num_layers):
            n_in  = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]

            # Initialisation He : sqrt(2 / n_in)
            W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            b = np.zeros((1, n_out))

            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):
        """
        Propagation avant
        X : (batch_size, 2)
        Retourne y_pred : (batch_size, 1)
        """
        self.activations = [X]   # stocke les sorties de chaque couche
        self.z_values    = []    # stocke les pré-activations (utile pour backprop)

        A = X
        for i in range(self.num_layers):
            Z = A @ self.weights[i] + self.biases[i]   # pré-activation
            self.z_values.append(Z)

            # ReLU pour les couches cachées, linéaire pour la sortie
            if i < self.num_layers - 1:
                A = relu(Z)
            else:
                A = Z   # couche de sortie : activation linéaire

            self.activations.append(A)

        return A   # y_pred


# ─────────────────────────────────────────────
# Test du forward pass
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # Chargement du dataset
    data = np.load("dataset_train.npy", allow_pickle=True)
    X = data[:, :2]   # (2000, 2)
    Y = data[:, 2:]   # (2000, 1)

    print(f"Dataset chargé : X={X.shape}, Y={Y.shape}")

    # Création du MLP
    mlp = MLP(layer_sizes=[2, 64, 64, 1])

    print(f"\nArchitecture : {mlp.layer_sizes}")
    for i in range(mlp.num_layers):
        print(f"  Couche {i+1} : W={mlp.weights[i].shape}, b={mlp.biases[i].shape}")

    # Forward pass sur tout le dataset
    y_pred = mlp.forward(X)
    loss = mse(y_pred, Y)

    print(f"\nForward pass OK")
    print(f"  Sortie shape : {y_pred.shape}")
    print(f"  MSE initiale : {loss:.6f}  (avant entraînement, valeur aléatoire attendue)")
