"""
TP RIC - Semaine 1 : Génération du Dataset
Université de Jijel - M1 Intelligence Artificielle

Fonction cible : f(x, y) = sin(sqrt(x² + y²)) + 0.5 * cos(2x + 2y)
"""

import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1. Fonction cible
# ─────────────────────────────────────────────
def f(x, y):
    return np.sin(np.sqrt(x**2 + y**2)) + 0.5 * np.cos(2*x + 2*y)


# ─────────────────────────────────────────────
# 2. Génération du dataset (2000 points aléatoires)
# ─────────────────────────────────────────────
np.random.seed(42)
N = 2000

x_raw = np.random.uniform(-5, 5, N)
y_raw = np.random.uniform(-5, 5, N)
z_raw = f(x_raw, y_raw)

print(f"Dataset généré : {N} points")
print(f"  x ∈ [{x_raw.min():.2f}, {x_raw.max():.2f}]")
print(f"  y ∈ [{y_raw.min():.2f}, {y_raw.max():.2f}]")
print(f"  z ∈ [{z_raw.min():.4f}, {z_raw.max():.4f}]")


# ─────────────────────────────────────────────
# 3. Normalisation
# ─────────────────────────────────────────────
# Entrées : min-max → [-1, 1]
X_norm = 2 * (x_raw - (-5)) / 10 - 1
Y_norm = 2 * (y_raw - (-5)) / 10 - 1

# Sortie : z-score
z_mean, z_std = z_raw.mean(), z_raw.std()
Z_norm = (z_raw - z_mean) / z_std

print(f"\nNormalisation :")
print(f"  Entrées → min-max [-1, 1]")
print(f"  Sortie  → z-score (μ={z_mean:.4f}, σ={z_std:.4f})")

# Sauvegarde
dataset = np.column_stack([X_norm, Y_norm, Z_norm])
np.save("dataset_train.npy", dataset)
np.save("norm_params.npy", {"z_mean": z_mean, "z_std": z_std})
print("Fichiers sauvegardés : dataset_train.npy, norm_params.npy")


# ─────────────────────────────────────────────
# 4. Visualisation
# ─────────────────────────────────────────────
# Grille dense pour la heatmap
res = 300
xg = np.linspace(-5, 5, res)
yg = np.linspace(-5, 5, res)
Xg, Yg = np.meshgrid(xg, yg)
Zg = f(Xg, Yg)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("TP RIC — Semaine 1 : f(x,y) = sin(√(x²+y²)) + 0.5·cos(2x+2y)", fontsize=12)

# Heatmap
ax = axes[0]
im = ax.imshow(Zg, extent=[-5, 5, -5, 5], origin='lower', cmap='plasma', aspect='auto')
plt.colorbar(im, ax=ax, label='z')
ax.set_title("Heatmap — vérité terrain")
ax.set_xlabel("x")
ax.set_ylabel("y")

# Scatter plot des 2000 points
ax = axes[1]
sc = ax.scatter(x_raw, y_raw, c=z_raw, cmap='plasma', s=5, alpha=0.6)
plt.colorbar(sc, ax=ax, label='z')
ax.set_title("Scatter plot — 2000 points générés")
ax.set_xlabel("x")
ax.set_ylabel("y")

plt.tight_layout()
plt.savefig("semaine1_visualisation.png", dpi=150, bbox_inches='tight')
plt.show()
print("Figure sauvegardée : semaine1_visualisation.png")