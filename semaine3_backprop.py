"""
TP RIC - Semaine 3 : Backpropagation
Université de Jijel - M1 Intelligence Artificielle

Construit sur la base de semaine2_mlp_forward.py
Ajoute : backward() et update_params()
"""

import numpy as np

# ─────────────────────────────────────────────
# Fonctions d'activation et de perte
# ─────────────────────────────────────────────

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


# ─────────────────────────────────────────────
# Classe MLP avec Backpropagation
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
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]
            W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            b = np.zeros((1, n_out))
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):
        """
        Propagation avant vectorisée
        X : (batch_size, 2)
        Retourne y_pred : (batch_size, 1)
        """
        self.activations = [X]
        self.z_values = []

        A = X
        for i in range(self.num_layers):
            Z = A @ self.weights[i] + self.biases[i]
            self.z_values.append(Z)
            if i < self.num_layers - 1:
                A = relu(Z)
            else:
                A = Z  # couche de sortie : activation linéaire
            self.activations.append(A)

        return A

    def backward(self, y_true):
        """
        Rétropropagation du gradient — entièrement vectorisée (calcul matriciel NumPy).
        Calcule les gradients dW et db pour chaque couche via la règle de la chaîne.

        Dérivation :
          - Perte MSE        : L = (1/N) * Σ (y_pred - y_true)²
          - dL/dy_pred       : (2/N) * (y_pred - y_true)
          - Couche sortie (linéaire) : delta = dL/dZ_out = dL/dy_pred
          - Couches cachées (ReLU)   : delta = (delta_next @ W_next.T) * relu'(Z)
          - Gradients poids  : dW = A_prev.T @ delta
          - Gradients biais  : db = sum(delta, axis=0)
        """
        batch_size = y_true.shape[0]

        self.dW = [None] * self.num_layers
        self.db = [None] * self.num_layers

        # Gradient de la perte MSE par rapport à la sortie
        delta = (2.0 / batch_size) * (self.activations[-1] - y_true)

        # Propagation en sens inverse, couche par couche
        for i in reversed(range(self.num_layers)):
            A_prev = self.activations[i]

            # Gradients des paramètres de la couche i
            self.dW[i] = A_prev.T @ delta                        # (n_in, n_out)
            self.db[i] = np.sum(delta, axis=0, keepdims=True)    # (1, n_out)

            # Gradient à propager vers la couche précédente
            if i > 0:
                delta = (delta @ self.weights[i].T) * relu_derivative(self.z_values[i - 1])

    def update_params(self, learning_rate):
        """Mise à jour des paramètres — descente de gradient (SGD)"""
        for i in range(self.num_layers):
            self.weights[i] -= learning_rate * self.dW[i]
            self.biases[i]  -= learning_rate * self.db[i]


# ─────────────────────────────────────────────
# Test de la backpropagation
# ─────────────────────────────────────────────

if __name__ == "__main__":
    data = np.load("dataset_train.npy", allow_pickle=True)
    X = data[:, :2]
    Y = data[:, 2:]

    print(f"Dataset chargé : X={X.shape}, Y={Y.shape}")

    np.random.seed(42)
    mlp = MLP(layer_sizes=[2, 64, 64, 1])

    # Forward
    y_pred = mlp.forward(X)
    loss_avant = mse(y_pred, Y)
    print(f"\nMSE avant update : {loss_avant:.6f}")

    # Backward + update
    mlp.backward(Y)
    mlp.update_params(learning_rate=0.001)

    # Vérification : la loss doit diminuer
    y_pred2 = mlp.forward(X)
    loss_apres = mse(y_pred2, Y)
    print(f"MSE après 1 update : {loss_apres:.6f}")
    print(f"Backpropagation OK ✓ (loss {'diminuée' if loss_apres < loss_avant else 'erreur !'})")

    # Vérification des shapes des gradients
    print(f"\nShapes des gradients :")
    for i in range(mlp.num_layers):
        print(f"  Couche {i+1} : dW={mlp.dW[i].shape}, db={mlp.db[i].shape}")
