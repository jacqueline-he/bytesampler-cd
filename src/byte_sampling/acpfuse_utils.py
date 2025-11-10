import torch
from typing import Tuple
import torch.nn.functional as F

# Super minor modifications from local_kl_acp_fuse.py

def kl(p: torch.Tensor, q: torch.Tensor):
    mask = torch.isneginf(p)
    return torch.where(
        mask, 0, F.kl_div(q, torch.where(mask, 0, p), reduction="none", log_target=True)
    ).sum(-1)

def interpolate(p: torch.Tensor, q: torch.Tensor, lam: torch.Tensor):
    lam = lam.unsqueeze(-1)
    mask = torch.isneginf(p) | torch.isneginf(q)
    # Stop the zero gradient from torch.where from hitting -inf
    p_safe, q_safe = torch.where(mask, 0, p), torch.where(mask, 0, q)
    mixed = torch.where(mask, -torch.inf, lam * p_safe + (1 - lam) * q_safe)
    res= torch.log_softmax(mixed, dim=-1)
    if res.dim() == 3:
        res = res.squeeze(1)
    return res

def dkl_dlam_bwd(p, q, lam):
    """
    Compute KL(interpolate(p,q,lam) || p) and its derivative w.r.t. lam.
    p, q: log-probs or logits (we treat them as tensors, not graph roots).
    lam: [B] tensor on same device.
    """
    # Detach p, q so we don't backprop into the model graph
    p = p.detach()
    q = q.detach()

    lam_leaf = lam.detach().clone().requires_grad_(True)  # fresh leaf
    pq = interpolate(p, q, lam_leaf)                      # [B, V] log-probs
    result = kl(pq, p)                                    # [B]

    # d/dlam result.sum()
    (dk,) = torch.autograd.grad(
        result.sum(),
        lam_leaf,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )
    return result.detach(), dk.detach()

def solve_optimization_batched(p, q, k_max, rounds=4, eps=1e-5):
    k_max = torch.full((p.shape[0],), float(k_max), device=p.device, dtype=p.dtype)
    lam = torch.full_like(k_max, 0.5)
    for i in range(rounds):
        k, dk = dkl_dlam_bwd(p, q, lam)
        lam = (lam + (k_max - k) / dk).clamp(0, 1)

    pq = interpolate(p, q, lam)
    
    # If lam=1 is close enough, just return q
    kq = kl(q, p)
    pq = torch.where(
        ((kq < k_max) | torch.isposinf(k_max)
    ).unsqueeze(-1), q, pq)
    
    # If even lam=0 is not close enough, just return p
    kpq = kl(pq, p)
    pq = torch.where((kpq > k_max).unsqueeze(-1), p, pq)

    w_c = lam 
    w_d = 1 - lam
    return w_c, w_d, pq

def safe_kl_pd_pc(log_pd: torch.Tensor, log_pc: torch.Tensor) -> torch.Tensor:
    """
    Compute KL(p_d || p_c) from log probs in a numerically safe way.
    log_pd, log_pc: [B, V]
    """
    p_d = torch.exp(log_pd)                # [B, V]
    diff = log_pd - log_pc                 # [B, V]

    # Mask out positions where p_d == 0 → contribution is exactly 0
    mask = p_d > 0
    diff = torch.where(mask, diff, torch.zeros_like(diff))
    p_d = torch.where(mask, p_d, torch.zeros_like(p_d))

    kl = (p_d * diff).sum(dim=-1)          # [B]
    return kl

def solve_optimization(
        clean_logits: torch.Tensor, # p_c logits 
        dirty_logits: torch.Tensor, # p_d logits
        k_radius: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve the optimization problem to find optimal weights.

        Args:
            clean_logits (torch.Tensor): Logits from clean model.
            dirty_logits (torch.Tensor): Logits from dirty model.
            k_radius (float): Radius for k-NAF guarantee.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Optimal b0, b1, b2 weights.
        """
        bc, bd = find_lambda(clean_logits, dirty_logits, k_radius)
        assert (abs(bc + bd - 1.0) < 1e-6).all(), "Weights must sum to 1"
        return bc, bd

def find_lambda(
    clean_logits: torch.Tensor,
    dirty_logits: torch.Tensor,
    k_radius: float,
    max_iter: int = 30,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = clean_logits.device
    B, V = clean_logits.shape

    # Convert logits to log-probs 
    log_pd = torch.log_softmax(dirty_logits, dim=-1) # [B, V]
    log_pc = torch.log_softmax(clean_logits, dim=-1) # [B, V]

    k_t = torch.as_tensor(k_radius, device=device, dtype=log_pd.dtype)


    # After computing k_t
    if verbose:
        print(f"[DEBUG] k_radius = {k_radius}, effective k_t = {k_t.tolist()}")


    # Corner case: if k_t <= 0, must sit exactly at p_c 
    mask_force_pc = k_t <= 0 # Force usage of p_c; we should never get here

    # Calculate KL divergence between p_d and p_c, and force usage of p_d if divergence is already large enough
    # KL: sum_x p_d(x) * (log p_d(x) - log p_c(x)) --> KL(p_d || p_c)
    # NEW: use safe_kl_pd_pc to avoid numerical instability
    KL_pd_pc = safe_kl_pd_pc(log_pd, log_pc) # [B]
    mask_use_pd = (KL_pd_pc <= k_t)

    # Initialize lambda bounds 
    lam_lo = torch.zeros(B, device=device, dtype=log_pd.dtype)
    lam_hi = torch.ones(B, device=device, dtype=log_pd.dtype)

    # Helper function to calculate KL divergence to p_c given a lambda 
    def kl_given_lambda(lam: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # w_d = 1/(1+lam), w_c = lam/(1+lam)
        w_d = 1.0 / (1.0 + lam)         # [B]
        w_c = 1.0 - w_d                 # [B]
        q_log = (w_d.unsqueeze(-1) * log_pd) + (w_c.unsqueeze(-1) * log_pc)   # Fusion of log-probs which is numerically stable: [B,V]
        # Re-normalizes q_log into proper log-probability distribution log_p
        log_p = q_log - torch.logsumexp(q_log, dim=-1, keepdim=False).unsqueeze(-1)

        # Compute KL(p || p_c) = sum p * (log p - log p_c)
        # NEW: use safe_kl_pd_pc to avoid numerical instability
        KL = safe_kl_pd_pc(log_p, log_pc) #[B]
        return KL, w_d, w_c


    # 1D grid search for lambda for the constraint KL(p(lam_lo) || p_c) <= k_t
    # Note that the objective min KL(p(lam_lo) || p_d) is implicitly satisfied by the logit fusion 
    with torch.no_grad():
        # We only need to search the samples where neither corner case applies 
        active = ~(mask_force_pc | mask_use_pd)
        if active.any():
            # Expand hi until KL <= k_t 
            for _ in range(max_iter):
                kl_hi, _, _ = kl_given_lambda(lam_hi)
                need_grow = active & (kl_hi > k_t)
                if not need_grow.any():
                    break
                lam_hi = torch.where(need_grow, (lam_hi * 2.0).clamp(max=1e6), lam_hi)
        
        # We are finding the largest lambda (lam_low) s.t. the KL divergence with this lambda <= k_t
        for _ in range(max_iter):
            lam_mid = 0.5 * (lam_lo + lam_hi)
            kl_mid, _, _ = kl_given_lambda(lam_mid)
            go_right = active & (kl_mid > k_t) # Need larger lambda 
            lam_lo = torch.where(go_right, lam_mid, lam_lo)
            lam_hi = torch.where(go_right, lam_hi, lam_mid)
    

    # Build final lambda (lam_lo) s.t. KL(p(lam_lo) || p_c) <= k_t. 
    lam = torch.zeros_like(lam_lo)
    # Force p_c 
    lam = torch.where(mask_force_pc, torch.full_like(lam, 1e6), lam)
    # # Force p_d 
    lam = torch.where(mask_use_pd, torch.zeros_like(lam), lam)

    mid = 0.5 * (lam_lo + lam_hi)
    lam = torch.where(~(mask_force_pc | mask_use_pd), mid, lam)

    # Map lambda to weights 
    w_d = (1.0 / (1.0 + lam)).unsqueeze(-1)   # b1 for dirty
    w_c = (1.0 - w_d)                         # b2 for clean 

    # Return weights
    return w_c, w_d

def get_fused_logp_from_weights(
    bc: torch.Tensor,  # [B,1] or [B]
    bd: torch.Tensor,  # [B,1] or [B]
    clean_logits: torch.Tensor, # [B,V]
    dirty_logits: torch.Tensor, # [B,V]
) -> Tuple[torch.Tensor, torch.Tensor]:

    clean_logits = torch.nan_to_num(clean_logits, neginf=-1e9)
    dirty_logits = torch.nan_to_num(dirty_logits, neginf=-1e9)

    log_pc = F.log_softmax(clean_logits, dim=-1)
    log_pd = F.log_softmax(dirty_logits, dim=-1)


    assert torch.isfinite(log_pc).all(), "Non-finite log_pc"
    assert torch.isfinite(log_pd).all(), "Non-finite log_pd"

    if bc.dim() == 2 and bc.size(1) == 1: bc = bc.squeeze(1)
    if bd.dim() == 2 and bd.size(1) == 1: bd = bd.squeeze(1)

    unnormalized_log_p = bd.unsqueeze(-1) * log_pd + bc.unsqueeze(-1) * log_pc

    # In case a row somehow becomes all -inf, fall back to clean or dirty
    all_neg_inf = torch.isneginf(unnormalized_log_p).all(dim=-1, keepdim=True)
    if all_neg_inf.any():
        # For those rows, just use clean model’s log probs as fallback
        unnormalized_log_p = torch.where(
            all_neg_inf,
            log_pc,  # or log_pd, or max(log_pc, log_pd)
            unnormalized_log_p,
        )

    log_p = unnormalized_log_p - torch.logsumexp(unnormalized_log_p, dim=-1, keepdim=True)
    assert torch.isfinite(log_p).all(), "Non-finite fused log_p"

    next_token_logits = bc.unsqueeze(-1) * clean_logits + bd.unsqueeze(-1) * dirty_logits
    return log_p, log_pc, next_token_logits
