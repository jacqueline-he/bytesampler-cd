import torch
from typing import Tuple

def solve_optimization(
    logits1: torch.Tensor,
    logits2: torch.Tensor,
    path_logprob1: torch.Tensor,
    path_logprob2: torch.Tensor,
    grid_size: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve the optimization problem to find optimal weights.

    Args:
        logits1 (torch.Tensor): Logits from model1.
        logits2 (torch.Tensor): Logits from model2.
        path_logprob1 (torch.Tensor): Accumulated log probabilities for model1.
        path_logprob2 (torch.Tensor): Accumulated log probabilities for model2.
        grid_size (int): Grid size for optimization.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Optimal b0, b1, b2 weights.
    """
    b0, b1, b2 = _optimize_grid(logits1, logits2, path_logprob1, path_logprob2, grid_size)
    return b0, 
    

def _optimize_grid(
        logits1: torch.Tensor,
        logits2: torch.Tensor,
        path_logprob1: torch.Tensor,
        path_logprob2: torch.Tensor,
        grid_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Optimize weights over a grid to minimize loss.

        Args:
            logits1 (torch.Tensor): Logits from model1.
            logits2 (torch.Tensor): Logits from model2.
            path_logprob1 (torch.Tensor): Accumulated log probabilities for model1.
            path_logprob2 (torch.Tensor): Accumulated log probabilities for model2.
            grid_size (int): Grid size for optimization.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Optimal b0, b1, b2 weights.
        """
        batch_size, vocab_size = logits1.shape
        device = logits1.device

        # Initialize b0 as zero tensor for all batch items
        b0 = torch.zeros(batch_size, 1, device=device)

        # Define the ranges
        first_range = torch.linspace(0, 2, steps=grid_size, device=device)
        second_range = torch.linspace(2, 10, steps=9, device=device)
        combined_range = torch.cat((first_range[:-1], second_range), dim=0)

        # Create a meshgrid for b1 and b2
        b1, b2 = torch.meshgrid(combined_range, combined_range, indexing="ij")
        b1 = b1.flatten()  # Shape: [grid_size^2]
        b2 = b2.flatten()  # Shape: [grid_size^2]

        # Expand b1 and b2 for batch and vocab_size
        b1_expanded = (
            b1.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, vocab_size)
        )  # Shape:  (batch_size, grid_size^2, vocab_size)
        b2_expanded = (
            b2.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, vocab_size)
        )  # Shape:  (batch_size, grid_size^2, vocab_size)

        # Expand logits
        logits1_expanded = logits1.unsqueeze(1)  # Shape: [batch_size, 1, vocab_size]
        logits2_expanded = logits2.unsqueeze(1)  # Shape: [batch_size, 1, vocab_size]

        # Compute loss for all combinations
        loss = cpfuse_objective(
            b0.unsqueeze(1),
            b1_expanded,
            b2_expanded,
            logits1_expanded,
            logits2_expanded,
            path_logprob1,
            path_logprob2,
        )  # Shape: [batch_size, grid_size^2]

        # Find the minimal loss and corresponding indices for b1 and b2
        _, min_idx = torch.min(loss, dim=1)
        optimal_b1 = b1[min_idx]
        optimal_b2 = b2[min_idx]

        # Optimal b0, b1, b2 are returned in shape [batch_size, 1]
        return b0, optimal_b1.unsqueeze(-1), optimal_b2.unsqueeze(-1)

def cpfuse_objective(
    b0: torch.Tensor,
    b1: torch.Tensor,
    b2: torch.Tensor,
    logits1: torch.Tensor,
    logits2: torch.Tensor,
    path_logprob1: torch.Tensor,
    path_logprob2: torch.Tensor,
) -> torch.Tensor:
    """Compute the objective function for optimization.

    Args:
        b0 (torch.Tensor): Weight scalar b0.
        b1 (torch.Tensor): Weight scalar b1.
        b2 (torch.Tensor): Weight scalar b2.
        logits1 (torch.Tensor): Logits from model1.
        logits2 (torch.Tensor): Logits from model2.
        path_logprob1 (torch.Tensor): Accumulated log probabilities for model1.
        path_logprob2 (torch.Tensor): Accumulated log probabilities for model2.

    Returns:
        torch.Tensor: Computed loss.
    """
    # Compute the combined log probabilities
    probs_log = _get_logits(b0, b1, b2, logits1, logits2)  # Shape: [batch_size, grid_size^2, vocab_size]
    probs_log = probs_log - torch.logsumexp(probs_log, dim=-1, keepdim=True)
    probs = probs_log.exp()  # Shape: [batch_size, grid_size^2, vocab_size]

    # Compute the expected log probabilities for each model
    loss1 = -(probs * logits1).sum(dim=-1) - path_logprob1  # Shape: [batch_size, grid_size^2]
    loss2 = -(probs * logits2).sum(dim=-1) - path_logprob2  # Shape: [batch_size, grid_size^2]

    # Expand path_logprob1 and path_logprob2 to match [batch_size, grid_size^2]
    if loss1.dim() < 2:
        loss1 = loss1.unsqueeze(1)
        loss2 = loss2.unsqueeze(1)

    total_loss = torch.max(loss1, loss2) + (probs * probs_log).sum(dim=-1)

    return total_loss  # Shape: [batch_size, grid_size^2]


def _get_logits(
        self,
        b0: torch.Tensor,
        b1: torch.Tensor,
        b2: torch.Tensor,
        logits1: torch.Tensor,
        logits2: torch.Tensor,
    ) -> torch.Tensor:
    """Compute combined logits.

    Args:
        b0 (torch.Tensor): Weight scalar b0.
        b1 (torch.Tensor): Weight scalar b1.
        b2 (torch.Tensor): Weight scalar b2.
        logits1 (torch.Tensor): Logits from model1.
        logits2 (torch.Tensor): Logits from model2.

    Returns:
        torch.Tensor: Combined logits.
    """
    # Check if we're in grid search (b1 has more than 2 dimensions) or main loop
    if b1.dim() > 2:
        # Grid search case
        # Ensure logits1 and logits2 have dimensions [batch_size, 1, vocab_size]
        if logits1.dim() == 2:
            logits1 = logits1.unsqueeze(1)  # [batch_size, 1, vocab_size]
            logits2 = logits2.unsqueeze(1)  # [batch_size, 1, vocab_size]
        # b0 has shape [batch_size, 1, 1], expand if necessary
        if b0.dim() == 2:
            b0 = b0.unsqueeze(1)  # [batch_size, 1, 1]
        combined_logits = b0 + b1 * logits1 + b2 * logits2  # [batch_size, grid_size^2, vocab_size]
    else:
        # Main loop case
        # Ensure b0, b1, b2 have shape [batch_size]
        b0 = b0.squeeze(-1)  # [batch_size]
        b1 = b1.squeeze(-1)  # [batch_size]
        b2 = b2.squeeze(-1)  # [batch_size]
        combined_logits = (
            b0.unsqueeze(-1) + b1.unsqueeze(-1) * logits1 + b2.unsqueeze(-1) * logits2
        )  # [batch_size, vocab_size]
    # combined_logits = combined_logits - torch.logsumexp(combined_logits, dim=-1, keepdim=True)
    return combined_logits