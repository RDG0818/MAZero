# core/mcts/sequential_search.py
import torch
from typing import List, Optional, Tuple
import numpy as np

from core.mcts import SampledMCTS, SearchOutput
from core.model import NetworkOutput, BaseNet
from core.config import BaseConfig


def sequential_search(
    config: BaseConfig,
    model: BaseNet,
    observation: torch.Tensor,
    legal_actions: np.ndarray,
    device: torch.device,
    add_noise: bool = False,
    sampled_tau: float = 1.0,
) -> List[SearchOutput]:
    """
    Run sequential, per-agent MCTS searches with conditioning via prefixes.

    Args:
        config:         BaseConfig containing num_agents, etc.
        model:          the BaseNet implementing initial_inference & recurrent_inference
        observation:    batched observations [B, ...]
        legal_actions:  np.ndarray of shape [B, N, A] indicating legal moves
        device:         torch.device for model inference
        add_noise:      whether to add root Dirichlet noise
        sampled_tau:    temperature for sampling the root policy

    Returns:
        A list of SearchOutput, one per agent in order, each containing
        the MCTS results when conditioning on previous agents.
    """
    # 1) Obtain the root network output once
    with torch.no_grad():
        root_out: NetworkOutput = model.initial_inference(observation)

    batch_size = observation.size(0)
    num_agents = config.num_agents

    # This will carry the fixed-actions prefix for the next agent
    prefix: Optional[np.ndarray] = None

    search_outputs: List[SearchOutput] = []

    # 2) Loop over agents sequentially
    for agent_id in range(num_agents):
        # Call the MCTS with or without conditioning
        if prefix is None:
            so = SampledMCTS(config).batch_search(
                model,
                root_out,
                legal_actions,
                device,
                add_noise=add_noise,
                sampled_tau=sampled_tau,
            )
        else:
            # prefix shape: [B, A]
            so = SampledMCTS(config).batch_search(
                model,
                root_out,
                legal_actions,
                device,
                add_noise=add_noise,
                sampled_tau=sampled_tau,
                sampled_actions_res=(prefix,)
            )

        search_outputs.append(so)

        # Extract this agent's marginal visit distribution to condition next
        # so.marginal_visit_count: np.ndarray [B, N, A]
        marg = so.marginal_visit_count[:, agent_id, :]
        # store as prefix for next iteration
        prefix = marg.copy()

    return search_outputs
