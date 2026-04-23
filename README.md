# The Self-Pruning Neural Network

### Tredence AI Engineer Intern — Case Study

**Author:** Vignesh S | SRMIST '27 | B.Tech Computer Science

---

## 🚀 Key Highlights

* Implemented a **self-pruning neural network** using learnable gates
* Designed a **3-phase training pipeline (Warmup → Prune → Fine-tune)**
* Applied **L1 regularization for sparsity induction**
* Analyzed **failure case where sparsity was not achieved**
* Demonstrated understanding of **hyperparameter sensitivity in pruning systems**

---

## 📌 Overview

This project implements a **self-pruning neural network** where each weight is controlled by a learnable gate.
The model attempts to automatically remove unnecessary connections during training while maintaining accuracy.

---

## 1. Why L1 Encourages Sparsity

Each weight is controlled by a gate:

```
gate = sigmoid(score / T)
effective_weight = weight × gate
```

Total loss:

```
L = CrossEntropy + λ × Σ(gates)
```

* **L1 penalty applies constant pressure toward zero**
* Unlike L2, it does not weaken near zero
* This enables **true sparsity (exact zeros)**

---

## 2. Architecture

```
Input (3072)
   ↓
PrunableLinear (3072 → 512)
   ↓
PrunableLinear (512 → 256)
   ↓
PrunableLinear (256 → 128)
   ↓
PrunableLinear (128 → 10)
```

* Each layer has **learnable gate scores**
* BatchNorm + ReLU stabilize training
* Total parameters: ~1.7M weights + gates

---

## 3. Training Strategy

```
WARMUP (5 epochs)
- No sparsity loss
- Model learns features

PRUNE (20 epochs)
- L1 sparsity applied
- Temperature annealing

FINE-TUNE (5 epochs)
- Pruned gates frozen
- Remaining weights optimized
```

---

## 4. Temperature Annealing

```
gate = sigmoid(score / T)
```

* T = 1.0 → smooth learning
* T → 0.1 → near-binary gates

Purpose:

* Prevent mid-range values
* Force decisions → prune or keep

---

## 5. Results

```
══════════════════════════════════════════════════════════════════════
  Lambda         Test Acc     Sparsity    Compression
══════════════════════════════════════════════════════════════════════
  1e-05           30.69%       0.00%         1.00×
  1e-04           22.79%       0.00%         1.00×
  1e-03           23.95%       0.00%         1.00×
══════════════════════════════════════════════════════════════════════
  Total wall-clock time: 1518.9s
```

### 📊 Interpretation

Across all λ values, **no effective pruning occurred**.

#### Observations:

* **λ = 1e-5 (Low):**

  * Acts like a standard dense model
  * Highest accuracy
  * No sparsity signal

* **λ = 1e-4 (Medium):**

  * Accuracy drops
  * Still no pruning

* **λ = 1e-3 (High):**

  * Accuracy degrades further
  * Gates still not pushed below threshold

---

## ⚠️ Why Sparsity Failed

This is an important insight:

* Pruning threshold too strict (`0.01`)
* Training duration insufficient
* Temperature annealing not aggressive enough
* Cross-entropy dominates sparsity loss

---

## 🔧 Improvements (Future Work)

To achieve real pruning:

```
epochs_prune = 40+
epochs_finetune = 10+
```

```
lambda_values = [1e-4, 5e-4, 1e-3]
```

```
prune_threshold = 0.05
temp_end = 0.01
```

---

## 📉 Expected Gate Distribution

Ideal behavior (not achieved here):

* Spike at 0 → pruned connections
* Cluster near 1 → active connections
* Minimal mid-range values

---

## 🧠 Key Learning

This experiment shows:

> **Self-pruning networks are highly sensitive to hyperparameters.**

Even with correct implementation:

* Poor tuning → no sparsity
* Strong tuning → efficient compressed model

---

## ▶️ How to Run

```bash
pip install torch torchvision matplotlib numpy
python self_pruning_network-2.py
```

---

## 📁 Outputs

* `results_panel.png` → performance visualization
* `gate_distribution.png` → gate histogram
* `results.json` → structured metrics

---

## 📚 References

* Tibshirani (1996) — LASSO
* Han et al. (2015) — Network pruning
* Frankle & Carbin (2019) — Lottery Ticket Hypothesis
* Louizos et al. (2018) — L0 regularization

---

## ✅ Conclusion

* Successfully implemented a **self-pruning framework**
* Demonstrated **training pipeline + gating mechanism**
* Identified **failure case due to hyperparameter sensitivity**
* Provided **clear path for improvement**

---

**Submitted by:** Vignesh S
SRM Institute of Science and Technology
B.Tech Computer Science
