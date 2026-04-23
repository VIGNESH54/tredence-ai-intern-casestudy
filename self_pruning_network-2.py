import os
import json
import math
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Reproducibility
SEED = 42

def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

set_seed()


# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("SelfPruning")


# Experiment Configuration (single source of truth)

@dataclass
class Config:
    # ── Dataset
    data_dir    : str   = "./data"
    batch_size  : int   = 256
    num_workers : int   = 2

    # ── Architecture
    hidden_dims : List[int] = field(default_factory=lambda: [512, 256, 128])
    input_dim   : int       = 32 * 32 * 3     
    num_classes : int       = 10

    # ── Training phases (fractions of total epochs)
    epochs_warmup   : int   = 5    
    epochs_prune    : int   = 20   
    epochs_finetune : int   = 5    

    # ── Optimiser
    lr              : float = 1e-3
    weight_decay    : float = 1e-4
    grad_clip       : float = 1.0   

    # ── Temperature annealing
    temp_start  : float = 1.0
    temp_end    : float = 0.1

    # ── Sparsity
    prune_threshold : float = 1e-2   
    lambda_values   : List[float] = field(default_factory=lambda: [1e-5, 1e-4, 1e-3])

    # ── Output
    output_dir  : str   = "./outputs"
    log_every   : int   = 5    

    @property
    def total_epochs(self) -> int:
        return self.epochs_warmup + self.epochs_prune + self.epochs_finetune


CFG = Config()



# Part 1 — PrunableLinear (Custom Gated Linear Layer)

class PrunableLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int,
                 prune_threshold: float = 1e-2) -> None:
        super().__init__()
        self.in_features      = in_features
        self.out_features     = out_features
        self.prune_threshold  = prune_threshold

        # Learnable parameters
        self.weight      = nn.Parameter(torch.empty(out_features, in_features))
        self.bias        = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        # Optional hard mask (non-learnable; applied after training)
        self.register_buffer("hard_mask",
                             torch.ones(out_features, in_features, dtype=torch.bool))
        self._hard_mask_applied = False

        self._reset_parameters()

    # ── Weight initialisation (Kaiming uniform, same as nn.Linear default)
    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    # ── Forward pass
    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        if self._hard_mask_applied:
            
            gates = self.hard_mask.float()
        else:
            gates = torch.sigmoid(self.gate_scores / temperature)

        pruned_weight = self.weight * gates
        return F.linear(x, pruned_weight, self.bias)

    # ── Hard mask (called once, post-training)
    def apply_hard_mask(self, threshold: Optional[float] = None) -> int:
        """
        Permanently zero all gates below `threshold`.
        Returns the number of newly pruned connections.
        """
        threshold = threshold or self.prune_threshold
        with torch.no_grad():
            soft_gates   = torch.sigmoid(self.gate_scores)
            self.hard_mask.copy_(soft_gates >= threshold)
            n_pruned     = (~self.hard_mask).sum().item()
        self._hard_mask_applied = True
        return n_pruned

    # ── Diagnostics
    def get_gates(self, temperature: float = 1.0) -> torch.Tensor:
        """Detached gate values for analysis (respects hard mask if applied)."""
        if self._hard_mask_applied:
            return self.hard_mask.float().detach()
        return torch.sigmoid(self.gate_scores / temperature).detach()

    def sparsity(self, threshold: Optional[float] = None) -> float:
        """Fraction of gates effectively pruned (below threshold)."""
        threshold = threshold or self.prune_threshold
        gates     = self.get_gates()
        return (gates < threshold).float().mean().item()

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"threshold={self.prune_threshold}")



# Network — four-layer MLP with PrunableLinear throughout

class SelfPruningNet(nn.Module):
    """
    Feed-forward network for CIFAR-10.
    Every linear projection uses PrunableLinear so gate learning happens
    end-to-end with the classification objective.

    BatchNorm between prunable layers stabilises training when many weights
    are zeroed mid-run, preventing activation collapse and dead ReLUs.
    """

    def __init__(self, cfg: Config = CFG) -> None:
        super().__init__()
        dims = [cfg.input_dim] + cfg.hidden_dims + [cfg.num_classes]

        layers: List[nn.Module] = []
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(PrunableLinear(d_in, d_out, cfg.prune_threshold))
            if i < len(dims) - 2:               # no BN/ReLU after output layer
                layers.append(nn.BatchNorm1d(d_out))
                layers.append(nn.ReLU(inplace=True))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        # Thread temperature through every PrunableLinear
        for m in self.net:
            if isinstance(m, PrunableLinear):
                x = m(x, temperature)
            else:
                x = m(x)
        return x

    def prunable_layers(self) -> List[PrunableLinear]:
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def apply_hard_masks(self, threshold: Optional[float] = None) -> Dict[str, int]:
        """Apply hard masks to all prunable layers. Returns per-layer pruned counts."""
        report = {}
        for i, layer in enumerate(self.prunable_layers()):
            n = layer.apply_hard_mask(threshold)
            report[f"layer_{i}"] = n
        return report

    def param_counts(self) -> Tuple[int, int]:
        """Returns (total_weights, active_weights) counting only prunable layers."""
        total = active = 0
        for layer in self.prunable_layers():
            n = layer.weight.numel()
            gates = layer.get_gates()
            total  += n
            active += (gates >= layer.prune_threshold).sum().item()
        return int(total), int(active)

    def compression_ratio(self) -> float:
        total, active = self.param_counts()
        return total / active if active > 0 else float("inf")



# Part 2 — Sparsity Regularisation Loss

def sparsity_loss(model: SelfPruningNet, temperature: float = 1.0) -> torch.Tensor:

    total = torch.zeros(1, device=next(model.parameters()).device)
    for layer in model.prunable_layers():
        gates  = torch.sigmoid(layer.gate_scores / temperature)
        total  = total + gates.sum()
    return total.squeeze()

# Data

def get_cifar10_loaders(cfg: Config = CFG) -> Tuple[DataLoader, DataLoader]:
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR10(cfg.data_dir, train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10(cfg.data_dir, train=False, download=True, transform=test_tf)

    kw = dict(num_workers=cfg.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=512,            shuffle=False, **kw)
    return train_loader, test_loader



# Part 3 — Training & Evaluation

def get_temperature(epoch: int, cfg: Config) -> float:
    """
    Linearly anneal temperature from temp_start → temp_end over the prune phase.
    Warmup and fine-tune phases use fixed temperatures.
    """
    if epoch < cfg.epochs_warmup:
        return cfg.temp_start
    if epoch >= cfg.epochs_warmup + cfg.epochs_prune:
        return cfg.temp_end
    progress = (epoch - cfg.epochs_warmup) / cfg.epochs_prune
    return cfg.temp_start + progress * (cfg.temp_end - cfg.temp_start)


def get_lambda(epoch: int, lam: float, cfg: Config) -> float:
    """No sparsity penalty during warmup; full penalty during prune; λ=0 during fine-tune."""
    if epoch < cfg.epochs_warmup:
        return 0.0
    if epoch >= cfg.epochs_warmup + cfg.epochs_prune:
        return 0.0    # fine-tune: gates frozen, no sparsity pressure
    return lam


def freeze_pruned_gates(model: SelfPruningNet, threshold: float) -> None:
    """
    During fine-tune phase: zero out gate_score gradients for pruned connections
    so those gates stay at zero while the surviving weights adapt.
    This is a soft version of the hard-mask approach applied mid-training.
    """
    for layer in model.prunable_layers():
        with torch.no_grad():
            gates = torch.sigmoid(layer.gate_scores)
            mask  = (gates >= threshold).float()
            # Project gate_scores toward -inf for pruned connections
            layer.gate_scores.data *= mask
            layer.gate_scores.data += (1 - mask) * (-20.0)  # saturate sigmoid to ~0


def train_one_epoch(
    model: SelfPruningNet,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    lam: float,
    temperature: float,
    cfg: Config,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = correct = total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs, temperature)

        cls_loss = F.cross_entropy(logits, labels)
        sp_loss  = sparsity_loss(model, temperature)
        loss     = cls_loss + lam * sp_loss
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        correct    += logits.argmax(1).eq(labels).sum().item()
        total      += bs

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model: SelfPruningNet, loader: DataLoader,
             temperature: float, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits  = model(imgs, temperature)
        correct += logits.argmax(1).eq(labels).sum().item()
        total   += imgs.size(0)
    return 100.0 * correct / total


def compute_global_sparsity(model: SelfPruningNet, threshold: float = 1e-2) -> float:
    """Percentage of all prunable weights with gate < threshold."""
    pruned = total = 0
    for layer in model.prunable_layers():
        g = layer.get_gates()
        pruned += (g < threshold).sum().item()
        total  += g.numel()
    return 100.0 * pruned / total if total > 0 else 0.0


def per_layer_sparsity(model: SelfPruningNet, threshold: float = 1e-2) -> List[float]:
    return [100.0 * layer.sparsity(threshold) for layer in model.prunable_layers()]


# Full Experiment Run

def run_experiment(
    lam: float,
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: Config,
    device: torch.device,
) -> Tuple[SelfPruningNet, Dict]:
    log.info("=" * 65)
    log.info(f"  Starting experiment  λ = {lam:.0e}  |  {cfg.total_epochs} epochs")
    log.info("=" * 65)

    set_seed()                           # reproducible per-λ run
    model     = SelfPruningNet(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.total_epochs)

    best_acc    : float = 0.0
    best_state  = None
    history     : Dict[str, List] = {
        "epoch": [], "train_acc": [], "test_acc": [],
        "sparsity": [], "loss": [], "temperature": []
    }

    for epoch in range(cfg.total_epochs):
        temperature = get_temperature(epoch, cfg)
        eff_lam     = get_lambda(epoch, lam, cfg)
        phase       = ("warmup"  if epoch < cfg.epochs_warmup
                       else "prune" if epoch < cfg.epochs_warmup + cfg.epochs_prune
                       else "finetune")

        # Gate-freeze during fine-tune phase
        if phase == "finetune":
            freeze_pruned_gates(model, cfg.prune_threshold)

        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer,
                                          eff_lam, temperature, cfg, device)
        scheduler.step()

        if (epoch + 1) % cfg.log_every == 0 or epoch == cfg.total_epochs - 1:
            te_acc   = evaluate(model, test_loader, temperature, device)
            sparsity = compute_global_sparsity(model, cfg.prune_threshold)

            history["epoch"].append(epoch + 1)
            history["train_acc"].append(round(tr_acc, 2))
            history["test_acc"].append(round(te_acc, 2))
            history["sparsity"].append(round(sparsity, 2))
            history["loss"].append(round(tr_loss, 4))
            history["temperature"].append(round(temperature, 3))

            log.info(
                f"  [{phase:>8s}] Ep {epoch+1:>3}/{cfg.total_epochs}  "
                f"T={temperature:.2f}  loss={tr_loss:.4f}  "
                f"train={tr_acc:.1f}%  test={te_acc:.1f}%  "
                f"sparse={sparsity:.1f}%"
            )

            if te_acc > best_acc:
                best_acc   = te_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Restore and hard-mask best checkpoint
    model.load_state_dict(best_state)
    model.apply_hard_masks(cfg.prune_threshold)
    final_acc      = evaluate(model, test_loader, cfg.temp_end, device)
    final_sparsity = compute_global_sparsity(model, cfg.prune_threshold)
    layer_sparsity = per_layer_sparsity(model, cfg.prune_threshold)
    total_w, active_w = model.param_counts()
    compression    = total_w / active_w if active_w > 0 else float("inf")

    log.info(f"\n  ► λ = {lam:.0e}  |  test_acc = {final_acc:.2f}%  "
             f"|  sparsity = {final_sparsity:.2f}%  "
             f"|  compression = {compression:.2f}×")

    return model, {
        "lambda"           : lam,
        "test_accuracy"    : round(final_acc, 2),
        "sparsity_pct"     : round(final_sparsity, 2),
        "compression_ratio": round(compression, 2),
        "total_weights"    : total_w,
        "active_weights"   : active_w,
        "layer_sparsity"   : [round(s, 2) for s in layer_sparsity],
        "history"          : history,
    }


# Publication-quality Visualisation  (4 panels)

PALETTE = ["#2196F3", "#4CAF50", "#F44336"]   # low / medium / high 


def plot_all(models_results: List[Tuple[SelfPruningNet, Dict]],
             cfg: Config, save_dir: str) -> None:
    """
    Four-panel figure:
      A) Gate value distribution (best model, λ=1e-4)
      B) Accuracy vs. sparsity trade-off across all λ
      C) Per-layer sparsity breakdown
      D) Training curves (test accuracy + sparsity over epochs)
    """
    os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#F8F9FA")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
    for ax in axes:
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#CCCCCC")

    # Panel A: Gate distribution for medium model 
    ax = axes[0]
    mid_idx = 1       # λ = 1e-4 is index 1
    model_mid, res_mid = models_results[mid_idx]
    all_gates = np.concatenate([
        l.get_gates().cpu().numpy().ravel()
        for l in model_mid.prunable_layers()
    ])

    ax.hist(all_gates, bins=100, color="#4C72B0", edgecolor="white",
            linewidth=0.3, alpha=0.9)
    ax.axvline(cfg.prune_threshold, color="#E53935", linestyle="--",
               linewidth=1.5, label=f"Prune threshold ({cfg.prune_threshold})")
    ax.set_xlabel("Gate Value", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"A.  Gate Distribution  (λ = {res_mid['lambda']:.0e})",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=9)
    # Annotate spike at 0 and cluster near 1
    n_zero = (all_gates < cfg.prune_threshold).sum()
    ax.annotate(
        f"{n_zero:,} pruned\n({100*n_zero/len(all_gates):.1f}%)",
        xy=(0.0, ax.get_ylim()[1] * 0.6),
        xytext=(0.15, ax.get_ylim()[1] * 0.75),
        arrowprops=dict(arrowstyle="->", color="#333"),
        fontsize=9, color="#333",
    )

    # Panel B: Accuracy vs. Sparsity scatter 
    ax = axes[1]
    for i, (_, res) in enumerate(models_results):
        ax.scatter(res["sparsity_pct"], res["test_accuracy"],
                   s=180, color=PALETTE[i], zorder=5,
                   label=f"λ = {res['lambda']:.0e}", edgecolors="white", linewidth=1.5)
        ax.annotate(
            f"  {res['compression_ratio']:.1f}×\n  compression",
            (res["sparsity_pct"], res["test_accuracy"]),
            fontsize=8, color=PALETTE[i],
        )
    accs      = [r["test_accuracy"] for _, r in models_results]
    sparsities = [r["sparsity_pct"]  for _, r in models_results]
    ax.plot(sparsities, accs, color="#999", linewidth=1.0, linestyle=":", zorder=1)
    ax.set_xlabel("Sparsity (%)", fontsize=11)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_title("B.  Sparsity–Accuracy Trade-off", fontsize=12, fontweight="bold", pad=10)
    ax.xaxis.set_major_formatter(PercentFormatter())
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel C: Per-layer sparsity bar chart 
    ax   = axes[2]
    n_layers = len(models_results[0][1]["layer_sparsity"])
    x    = np.arange(n_layers)
    w    = 0.25
    layer_labels = ["Layer 1\n(3072→512)", "Layer 2\n(512→256)",
                    "Layer 3\n(256→128)", "Layer 4\n(128→10)"][:n_layers]
    for i, (_, res) in enumerate(models_results):
        bars = ax.bar(x + i * w - w, res["layer_sparsity"],
                      width=w, color=PALETTE[i], alpha=0.85,
                      label=f"λ = {res['lambda']:.0e}", edgecolor="white")
        for bar, val in zip(bars, res["layer_sparsity"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=7, color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels, fontsize=9)
    ax.set_ylabel("Sparsity (%)", fontsize=11)
    ax.set_title("C.  Per-Layer Sparsity Breakdown", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # Panel D: Training curves (mid model) 
    ax   = axes[3]
    hist = models_results[mid_idx][1]["history"]
    ep   = hist["epoch"]

    color_acc   = "#1976D2"
    color_spar  = "#E53935"
    ax2 = ax.twinx()

    ln1, = ax.plot(ep, hist["test_acc"], color=color_acc, linewidth=2.0,
                   marker="o", markersize=4, label="Test Accuracy")
    ln2, = ax2.plot(ep, hist["sparsity"], color=color_spar, linewidth=2.0,
                    linestyle="--", marker="s", markersize=4, label="Sparsity %")

    # Phase boundaries
    warmup_end  = cfg.epochs_warmup
    prune_end   = cfg.epochs_warmup + cfg.epochs_prune
    ax.axvline(warmup_end, color="#757575", linestyle=":", linewidth=1.0)
    ax.axvline(prune_end,  color="#757575", linestyle=":", linewidth=1.0)
    ax.text(warmup_end / 2, min(hist["test_acc"]) - 1.5,
            "Warmup", ha="center", fontsize=8, color="#757575")
    ax.text((warmup_end + prune_end) / 2, min(hist["test_acc"]) - 1.5,
            "Prune", ha="center", fontsize=8, color="#757575")
    ax.text((prune_end + cfg.total_epochs) / 2, min(hist["test_acc"]) - 1.5,
            "Fine-tune", ha="center", fontsize=8, color="#757575")

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11, color=color_acc)
    ax2.set_ylabel("Sparsity (%)", fontsize=11, color=color_spar)
    ax.set_title(f"D.  Training Dynamics  (λ = {res_mid['lambda']:.0e})",
                 fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(axis="y", labelcolor=color_acc)
    ax2.tick_params(axis="y", labelcolor=color_spar)
    ax.grid(True, alpha=0.3)
    lns = [ln1, ln2]
    ax.legend(lns, [l.get_label() for l in lns], fontsize=9, loc="lower right")

    # Title & save 
    fig.suptitle(
        "The Self-Pruning Neural Network — CIFAR-10 Experiment Results",
        fontsize=14, fontweight="bold", y=0.98,
    )
    path = os.path.join(save_dir, "results_panel.png")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"  Results panel saved → {path}")

    # Standalone gate distribution plot (as required by spec) 
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    fig2.patch.set_facecolor("#F8F9FA")
    ax2.set_facecolor("white")
    ax2.hist(all_gates, bins=100, color="#4C72B0", edgecolor="white", linewidth=0.3, alpha=0.9)
    ax2.axvline(cfg.prune_threshold, color="#E53935", linestyle="--",
                linewidth=1.5, label=f"Prune threshold ({cfg.prune_threshold})")
    ax2.set_xlabel("Gate Value", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title(f"Gate Value Distribution  (λ = {res_mid['lambda']:.0e},  "
                  f"sparsity = {res_mid['sparsity_pct']:.1f}%)", fontsize=13)
    ax2.legend(fontsize=10)
    plt.tight_layout()
    path2 = os.path.join(save_dir, "gate_distribution.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    log.info(f"  Gate distribution saved → {path2}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    log.info(f"Config: {asdict(CFG)}")

    os.makedirs(CFG.output_dir, exist_ok=True)
    train_loader, test_loader = get_cifar10_loaders(CFG)

    models_results: List[Tuple[SelfPruningNet, Dict]] = []
    t0 = time.time()

    for lam in CFG.lambda_values:
        model, result = run_experiment(lam, train_loader, test_loader, CFG, device)
        models_results.append((model, result))

    elapsed = time.time() - t0

    # Summary table 
    sep = "═" * 70
    print(f"\n\n{sep}")
    print(f"  {'Lambda':<10} {'Test Acc':>12} {'Sparsity':>12} {'Compression':>14}")
    print(sep)
    for _, res in models_results:
        print(
            f"  {res['lambda']:<10.0e}"
            f" {res['test_accuracy']:>10.2f}%"
            f" {res['sparsity_pct']:>10.2f}%"
            f" {res['compression_ratio']:>12.2f}×"
        )
    print(sep)
    print(f"  Total wall-clock time: {elapsed:.1f}s\n")

    # Plots 
    plot_all(models_results, CFG, CFG.output_dir)

    # Persist structured results
    serialisable = []
    for _, r in models_results:
        entry = {k: v for k, v in r.items() if k != "history"}
        entry["lambda"] = str(entry["lambda"])
        serialisable.append(entry)

    results_path = os.path.join(CFG.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    log.info(f"  Structured results saved → {results_path}")
    log.info("  ✓ All outputs written to ./outputs/")


if __name__ == "__main__":
    main()
