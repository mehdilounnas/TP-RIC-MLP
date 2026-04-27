"""
TP RIC - Semaine 4 : Entraînement et Visualisation
Université de Jijel - M1 Intelligence Artificielle

Dépend de : semaine3_backprop.py (importe la classe MLP)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from semaine3_backprop import MLP, mse


# ─────────────────────────────────────────────
# Fonction cible originale (pour visualisation)
# ─────────────────────────────────────────────

def f(x, y):
    return np.sin(np.sqrt(x**2 + y**2)) + 0.5 * np.cos(2*x + 2*y)


# ─────────────────────────────────────────────
# Fonction d'entraînement
# ─────────────────────────────────────────────

def train(mlp, X, Y, epochs=800, batch_size=64, learning_rate=0.001):
    """
    Entraînement par mini-batch SGD.
    Retourne l'historique de la loss par époque.
    """
    N = X.shape[0]
    loss_history = []

    for epoch in range(epochs):
        # Mélange aléatoire
        indices = np.random.permutation(N)
        X_sh = X[indices]
        Y_sh = Y[indices]

        epoch_loss = 0.0
        num_batches = 0

        # Mini-batchs (pas de boucle sur les exemples individuels)
        for start in range(0, N, batch_size):
            X_batch = X_sh[start:start + batch_size]
            Y_batch = Y_sh[start:start + batch_size]

            y_pred = mlp.forward(X_batch)
            mlp.backward(Y_batch)
            mlp.update_params(learning_rate)

            epoch_loss += mse(y_pred, Y_batch)
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)

        if (epoch + 1) % 100 == 0:
            print(f"  Époque {epoch+1:4d}/{epochs} | Loss : {avg_loss:.6f}")

    return loss_history


# ─────────────────────────────────────────────
# Chargement des données
# ─────────────────────────────────────────────

print("=" * 50)
print("  TP RIC — Semaine 4 : Entraînement + Visualisation")
print("=" * 50)

data = np.load("dataset_train.npy", allow_pickle=True)
X = data[:, :2]
Y = data[:, 2:]

norm_params = np.load("norm_params.npy", allow_pickle=True).item()
z_mean = norm_params["z_mean"]
z_std  = norm_params["z_std"]

print(f"\nDataset chargé : X={X.shape}, Y={Y.shape}")
print(f"Normalisation sortie : μ={z_mean:.4f}, σ={z_std:.4f}")


# ─────────────────────────────────────────────
# Entraînement
# ─────────────────────────────────────────────

np.random.seed(42)
mlp = MLP(layer_sizes=[2, 64, 64, 1])

print(f"\nArchitecture : {mlp.layer_sizes}")
print(f"Hyperparamètres :")
print(f"  learning_rate = 0.001")
print(f"  batch_size    = 64")
print(f"  epochs        = 800")
print(f"\nEntraînement...")

loss_history = train(mlp, X, Y, epochs=800, batch_size=64, learning_rate=0.001)

print(f"\nLoss initiale : {loss_history[0]:.6f}")
print(f"Loss finale   : {loss_history[-1]:.6f}")
print(f"Réduction     : {(1 - loss_history[-1]/loss_history[0])*100:.1f}%")


# ─────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────

# Grille de test sur [-5, 5]
res = 100
xg = np.linspace(-5, 5, res)
yg = np.linspace(-5, 5, res)
Xg, Yg = np.meshgrid(xg, yg)

# Normalisation de la grille (identique à semaine 1)
Xg_norm = 2 * (Xg - (-5)) / 10 - 1
Yg_norm = 2 * (Yg - (-5)) / 10 - 1
grid_input = np.column_stack([Xg_norm.ravel(), Yg_norm.ravel()])

# Prédiction + dé-normalisation
Z_pred_norm = mlp.predict(grid_input) if hasattr(mlp, 'predict') else mlp.forward(grid_input)
Z_pred = Z_pred_norm.reshape(res, res) * z_std + z_mean
Z_true = f(Xg, Yg)

# ── Figure 1 : Loss + Heatmaps côte à côte ───
fig1 = plt.figure(figsize=(18, 5))
fig1.suptitle("TP RIC — Semaine 4 : Entraînement MLP [2, 64, 64, 1]", fontsize=13)

ax1 = fig1.add_subplot(1, 3, 1)
ax1.plot(loss_history, color='steelblue', linewidth=1.5)
ax1.set_title("Courbe de perte (MSE)")
ax1.set_xlabel("Époque")
ax1.set_ylabel("MSE Loss")
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)
ax1.annotate(f"Final: {loss_history[-1]:.5f}",
             xy=(len(loss_history)-1, loss_history[-1]),
             xytext=(len(loss_history)*0.5, loss_history[0]*0.3),
             arrowprops=dict(arrowstyle='->', color='red'),
             color='red', fontsize=9)

vmin, vmax = Z_true.min(), Z_true.max()

ax2 = fig1.add_subplot(1, 3, 2)
im2 = ax2.imshow(Z_true, extent=[-5,5,-5,5], origin='lower',
                 cmap='plasma', aspect='auto', vmin=vmin, vmax=vmax)
plt.colorbar(im2, ax=ax2, label='z')
ax2.set_title("Vérité terrain\nf(x,y) = sin(√(x²+y²)) + 0.5·cos(2x+2y)")
ax2.set_xlabel("x"); ax2.set_ylabel("y")

ax3 = fig1.add_subplot(1, 3, 3)
im3 = ax3.imshow(Z_pred, extent=[-5,5,-5,5], origin='lower',
                 cmap='plasma', aspect='auto', vmin=vmin, vmax=vmax)
plt.colorbar(im3, ax=ax3, label='z')
ax3.set_title(f"Prédiction MLP\n(MSE finale = {loss_history[-1]:.5f})")
ax3.set_xlabel("x"); ax3.set_ylabel("y")

plt.tight_layout()
plt.savefig("semaine4_resultat.png", dpi=150, bbox_inches='tight')
plt.show()
print("Figure sauvegardée : semaine4_resultat.png")

# ── Figure 2 : Surfaces 3D côte à côte ────────
fig2 = plt.figure(figsize=(14, 5))
fig2.suptitle("Comparaison 3D : Vérité terrain vs Prédiction MLP", fontsize=12)

ax_t = fig2.add_subplot(1, 2, 1, projection='3d')
ax_t.plot_surface(Xg, Yg, Z_true, cmap='plasma', alpha=0.85)
ax_t.set_title("Vérité terrain")
ax_t.set_xlabel("x"); ax_t.set_ylabel("y"); ax_t.set_zlabel("z")

ax_p = fig2.add_subplot(1, 2, 2, projection='3d')
ax_p.plot_surface(Xg, Yg, Z_pred, cmap='plasma', alpha=0.85)
ax_p.set_title("Prédiction MLP")
ax_p.set_xlabel("x"); ax_p.set_ylabel("y"); ax_p.set_zlabel("z")

plt.tight_layout()
plt.savefig("semaine4_3d_comparaison.png", dpi=150, bbox_inches='tight')
plt.show()
print("Figure 3D sauvegardée : semaine4_3d_comparaison.png")
