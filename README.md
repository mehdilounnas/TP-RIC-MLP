# TP RIC — Cartographie d'une Fonction Mystère
**Université de Jijel | Département d'Informatique | M1 Intelligence Artificielle**

Implementation of a complete **Multilayer Perceptron (MLP) from scratch** using only NumPy, trained to approximate the mathematical surface:

$$f(x, y) = \sin\left(\sqrt{x^2 + y^2}\right) + 0.5 \cdot \cos(2x + 2y)$$

---

## 📁 Project Structure

| File | Week | Description |
|---|---|---|
| `semaine1_dataset.py` | Week 1 | Dataset generation, normalization, 3D visualization |
| `semaine2_mlp_forward.py` | Week 2 | MLP class with He initialization + forward pass |
| `semaine3_backprop.py` | Week 3 | Backpropagation (vectorized) + gradient update |
| `semaine4_training.py` | Week 4 | Mini-batch SGD training + final visualizations |
| `rapport_mlp.pdf` | Final | Full report with visualizations and analysis |
| `dataset_train.npy` | — | Generated training dataset (2000 points) |
| `norm_params.npy` | — | Normalization parameters (μ, σ) |

---

## 🧠 Architecture

```
Input (2) → Hidden (64) → Hidden (64) → Output (1)
```

- **Activation:** ReLU for hidden layers, Linear for output
- **Weight init:** He initialization
- **Loss:** Mean Squared Error (MSE)
- **Optimizer:** Mini-batch SGD

---

## 📊 Results

| Architecture | Activation | MSE Final | Result |
|---|---|---|---|
| [2, 64, 64, 1] | ReLU | **0.376** | ✅ Good approximation |
| [2, 4, 1] | ReLU | 0.712 | ⚠️ Underfitting |
| [2, 64, 64, 1] | Linear | 1.000 | ❌ Total failure |

Training: **800 epochs**, batch size **64**, learning rate **0.001** → **62.9% MSE reduction**

---

## 🚀 How to Run

```bash
# Install dependency
pip install numpy matplotlib

# Run week by week
python semaine1_dataset.py    # generates dataset + visualizations
python semaine2_mlp_forward.py # tests forward pass
python semaine3_backprop.py    # tests backpropagation
python semaine4_training.py    # full training + final plots
```

> ⚠️ Run `semaine1_dataset.py` first — it generates `dataset_train.npy` and `norm_params.npy` required by the other scripts.

---

## 👤 Author
**Lounnas Mehdi** — M1 Intelligence Artificielle, Université de Jijel (2025/2026)
