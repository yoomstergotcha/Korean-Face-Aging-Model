import torch
import torch.nn.functional as F


# =========================
# Basic losses
# =========================
def l1_loss(x, y):
    return torch.mean(torch.abs(x - y))


def cosine_dist(a, b, eps=1e-8):
    a = F.normalize(a, dim=1, eps=eps)
    b = F.normalize(b, dim=1, eps=eps)
    return 1.0 - torch.sum(a * b, dim=1).mean()


def cosine_dist_per_sample(a, b, eps=1e-8):
    a = F.normalize(a, dim=1, eps=eps)
    b = F.normalize(b, dim=1, eps=eps)
    return 1.0 - torch.sum(a * b, dim=1)


# =========================
# GAN (hinge loss)
# =========================
def d_hinge_loss(real_logits, fake_logits):
    return (
        torch.relu(1.0 - real_logits).mean() +
        torch.relu(1.0 + fake_logits).mean()
    )


def g_hinge_loss(fake_logits):
    return -fake_logits.mean()


# =========================
# Identity gating weight
# =========================
def identity_weight(age_gap, tau=15.0):
    """
    age_gap : (B,)
    small age gap → stronger identity constraint
    """
    return torch.exp(-age_gap / tau)


# =========================
# Age binning (for prototypes)
# =========================
def age_to_bin(age, bins=(0,10,20,30,40,50,60,70,80)):
    """
    age : (B,1) or (B,)
    returns LongTensor (B,)
    """
    age = age.view(-1)
    out = torch.zeros_like(age, dtype=torch.long)

    for i in range(len(bins) - 1):
        mask = (age >= bins[i]) & (age < bins[i+1])
        out[mask] = i

    out[age >= bins[-1]] = len(bins) - 2
    return out
