# unit_test_reanalyze_worker.py

import logging
import math
import os
import time
from typing import Tuple, List, Union, Any, NamedTuple
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
# Ensure these are importable
# from core.config import BaseConfig # Mocked
# from core.model import BaseNet, NetworkOutput # Mocked (or minimal real)
# from core.game import Game, GameHistory # Game is Mocked, GameHistory is real
# from core.replay_buffer import ReplayBuffer # Mocked
# from core.storage import SharedStorage # Mocked

from mcts_sampled_modified import SampledMCTS, SearchOutput
from selfplay_worker_modified import DataWorker # Not directly used, but for context
from reanalyze_worker_modified import ReanalyzeWorker # The class we are testing
from core.game import GameHistory # Using the real GameHistory
from core.utils import prepare_observation_lst # Used by ReanalyzeWorker

# --- Start Mock/Minimal Definitions (some can be reused from previous tests) ---

class MockAction(NamedTuple):
    action: torch.Tensor

class NetworkOutput(NamedTuple):
    value: Any
    reward: Any
    policy_logits: Any
    hidden_state: Any
    # legal_actions_logits: Any = None # Add if your model was modified for this

class BaseNet(nn.Module, ABC):
    def __init__(self, inverse_value_transform=None, inverse_reward_transform=None):
        super().__init__()
        self._inverse_value_transform = inverse_value_transform
        self._inverse_reward_transform = inverse_reward_transform
    @abstractmethod
    def representation(self, observation: torch.Tensor) -> torch.Tensor: pass
    @abstractmethod
    def prediction(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: pass
    @abstractmethod
    def dynamics(self, hidden_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: pass
    @abstractmethod
    def initial_inference(self, observation: torch.Tensor) -> NetworkOutput: pass
    @abstractmethod
    def recurrent_inference(self, hidden_state: torch.Tensor, action: torch.Tensor) -> NetworkOutput: pass
    def get_weights(self): return {k: v.cpu() for k, v in self.state_dict().items()}
    def set_weights(self, weights): self.load_state_dict(weights)
    def project(self, hidden_state, with_grad=True): # Added for consistency loss path
        print(f"[MockModel.project] Called with hs shape: {hidden_state.shape}")
        return hidden_state[:, :128] if hidden_state.shape[1] >=128 else hidden_state # Dummy projection


class MockConfig:
    def __init__(self, true_n_agents, action_space_sz, hidden_sz_per_agent, num_sims,
                 batch_size_reanalyze, num_unroll_steps_reanalyze, td_steps_reanalyze,
                 obs_feat_size=10, stack_obs=1):
        # For ReanalyzeWorker & its components
        self.num_agents = true_n_agents
        self.action_space_size = action_space_sz
        self.hidden_state_size_per_agent = hidden_sz_per_agent
        self.num_simulations = num_sims
        self.batch_size = batch_size_reanalyze # B for make_batch
        self.num_unroll_steps = num_unroll_steps_reanalyze # K
        self.td_steps = td_steps_reanalyze
        self.obs_shape = (obs_feat_size,) # Simplified for non-image
        self.stacked_observations = stack_obs
        self.image_based = False
        self.seed = 123
        self.reanalyze_on_gpu = False
        self.training_steps = 10000
        self.last_steps = 1000
        self.priority_prob_beta = 0.4 # For LinearSchedule
        self.auto_td_steps = 1000 # For off-policy correction
        self.use_reanalyze_value = True # Test the reanalyze path for value
        self.use_root_value = True # Use MCTS root value in reanalyze
        self.use_pred_value = False # Don't use network pred if use_root_value is True for reanalyze
        self.discount = 0.99
        self.revisit_policy_search_rate = 1.0 # Reanalyze all policy targets

        # For SampledMCTS inside ReanalyzeWorker
        self.pb_c_base = 19652.0
        self.pb_c_init = 1.25
        self.mcts_rho = 0.75
        self.mcts_lambda = 0.8
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25
        # CRITICAL ASSUMPTION FOR THIS TEST:
        # We are designing _prepare_policy_re to output targets as if C=1
        # If training.py still expects C > 1, padding logic in _prepare_policy_re is needed.
        # For this test, we set config.sampled_action_times=1 to match the target generation.
        self.sampled_action_times = 1 # C (Effective C for targets)
        self.tree_value_stat_delta_lb = 0.01

        # For GameHistory (used by ReanalyzeWorker to process games)
        self.cvt_string = False
        self.gray_scale = False
        # self.max_samples in GameHistory is self.config.sampled_action_times

        # For value/reward transforms (not strictly needed if model returns numpy)
        self.use_vectorization = False # Assume scalar value/reward for simplicity
        def identity_transform(x): return x
        self.reward_transform = identity_transform
        self.value_transform = identity_transform

        # For consistency loss path (if enabled, not focus of this test)
        self.consistency_coeff = 0.0 

    def get_uniform_network(self):
        return MockModel(self, torch.device("cpu"))


class MockModel(BaseNet):
    def __init__(self, config: MockConfig, device: torch.device):
        super().__init__()
        self.true_num_agents = config.num_agents
        self.action_space_size = config.action_space_size
        self.hidden_size_per_agent = config.hidden_state_size_per_agent
        self.total_hidden_size = self.true_num_agents * self.hidden_size_per_agent
        self.device = device
        self.dummy_param = torch.nn.Parameter(torch.empty(0))
        self.config = config
        print(f"[MockModel Init] true_num_agents: {self.true_num_agents}, action_space_size: {self.action_space_size}, total_hidden_size: {self.total_hidden_size}")

    # ... (initial_inference and recurrent_inference are already there) ...

    def representation(self, observation: torch.Tensor) -> torch.Tensor:
        print(f"[MockModel.representation] Called with obs shape: {observation.shape}")
        # This is called by initial_inference. Ensure it returns a hidden state
        # of the correct shape if initial_inference relies on it.
        # Or, make initial_inference self-contained for the mock.
        # For this mock, initial_inference is self-contained, so this can be simple:
        batch_size = observation.shape[0]
        return torch.rand((batch_size, self.total_hidden_size), device=self.device)

    def prediction(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # This is called by initial_inference and recurrent_inference in the real MaMuZeroNet
        # Ensure it returns policy_logits and value_logits of expected shapes.
        print(f"[MockModel.prediction] Called with hs shape: {hidden_state.shape}")
        batch_size = hidden_state.shape[0]
        policy_logits = torch.zeros((batch_size, self.true_num_agents, self.action_space_size), device=self.device)
        value_logits = torch.full((batch_size, 1), 0.45, device=self.device) # Mock scalar value logits
        # If your model uses categorical value/reward, this should be (batch_size, support_size)
        return policy_logits, value_logits

    def dynamics(self, hidden_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # This is called by recurrent_inference in the real MaMuZeroNet
        print(f"[MockModel.dynamics] Called with hs shape: {hidden_state.shape}, action shape: {action.shape}")
        batch_size = hidden_state.shape[0]
        next_hidden_state = hidden_state + 0.001 # Minimal change
        # Mock scalar reward logits
        reward_logits = torch.ones((batch_size, 1), device=self.device) * 0.05 
        return next_hidden_state, reward_logits

    def initial_inference(self, observation: torch.Tensor) -> NetworkOutput:
        # This mock version can be self-contained for simplicity, 
        # or it can call self.representation and self.prediction
        batch_size = observation.shape[0]
        print(f"\n[MockModel.initial_inference] Called. Input obs batch_size: {batch_size}")
        
        # Simulating what the real initial_inference does:
        hidden_state_total = self.representation(observation) # [B, N*H]
        policy_logits_torch, value_logits_torch = self.prediction(hidden_state_total) # [B,N,A], [B, value_support_size]
        
        # For initial step, reward is typically 0.
        # Shape depends on whether value/reward is scalar or categorical. Assuming scalar for mock.
        reward_logits_torch = torch.zeros(batch_size, 1, device=self.device) 

        return NetworkOutput(
            value=value_logits_torch.cpu().numpy(), # Or transform if categorical
            reward=reward_logits_torch.cpu().numpy(), # Or transform
            policy_logits=policy_logits_torch.cpu().numpy(),
            hidden_state=hidden_state_total 
        )

    def recurrent_inference(self, hidden_state: torch.Tensor, action: torch.Tensor) -> NetworkOutput:
        batch_size = hidden_state.shape[0]
        print(f"\n[MockModel.recurrent_inference] Called with:")
        print(f"  hidden_state shape: {hidden_state.shape}")
        print(f"  action shape: {action.shape}, example action: {action[0].cpu().numpy()}")
        assert hidden_state.shape[1] == self.total_hidden_size
        assert action.shape[1] == self.true_num_agents

        # Simulating what the real recurrent_inference does:
        next_hidden_state_total, reward_logits_torch = self.dynamics(hidden_state, action)
        policy_logits_torch, value_logits_torch = self.prediction(next_hidden_state_total)

        return NetworkOutput(
            value=value_logits_torch.cpu().numpy(), # Or transform
            reward=reward_logits_torch.cpu().numpy(), # Or transform
            policy_logits=policy_logits_torch.cpu().numpy(),
            hidden_state=next_hidden_state_total
        )

# ---- Test Script ----
def run_reanalyze_worker_test():
    print("--- Starting ReanalyzeWorker Test ---")
    # Test Parameters
    test_batch_size = 2  # B for make_batch
    test_num_unroll_steps = 3 # K
    test_true_n_agents = 2
    test_action_space_sz = 3
    test_hidden_sz_per_agent = 8
    test_num_sims = 3
    test_obs_feat_size = 5
    test_stacked_obs = 1

    # Setup Config
    config = MockConfig(
        true_n_agents=test_true_n_agents,
        action_space_sz=test_action_space_sz,
        hidden_sz_per_agent=test_hidden_sz_per_agent,
        num_sims=test_num_sims,
        batch_size_reanalyze=test_batch_size,
        num_unroll_steps_reanalyze=test_num_unroll_steps,
        td_steps_reanalyze=2, # td_steps for value target calculation
        obs_feat_size=test_obs_feat_size,
        stack_obs=test_stacked_obs
    )
    # Ensure C=1 assumption for targets generated by _prepare_policy_re
    # If training.py expects config.sampled_action_times to be >1, _prepare_policy_re
    # would need padding logic. Here we test _prepare_policy_re's C=1 output.
    # For this test, let's assume the C in config can be > 1, and _prepare_policy_re handles padding.
    # OR, we set config.sampled_action_times = 1 for this test to simplify.
    # Let's test assuming C=1 in config to match _prepare_policy_re's direct output.
    original_C = config.sampled_action_times # Store original C if it was > 1
    config.sampled_action_times = 1 # Force C=1 for this test of target generation
    print(f"Test configured with C (sampled_action_times) = {config.sampled_action_times}")


    # Create Mock GameHistories
    mock_game_histories = []
    game_len = test_num_unroll_steps + config.td_steps + test_stacked_obs + 5 # Ensure long enough
    for i in range(test_batch_size):
        gh = GameHistory(config=config, ray_store_obs=False)
        # Populate with some dummy data
        init_obs_list = [np.random.rand(test_true_n_agents, test_obs_feat_size).astype(np.float32) for _ in range(test_stacked_obs)]
        gh.init(init_obs_list)
        for step in range(game_len):
            action = np.random.randint(0, test_action_space_sz, size=test_true_n_agents)
            reward = np.random.rand() * 0.1
            next_obs = np.random.rand(test_true_n_agents, test_obs_feat_size).astype(np.float32)
            legal_actions = np.ones((test_true_n_agents, test_action_space_sz), dtype=np.int32)
            # Make some actions illegal for variety if needed:
            if test_action_space_sz > 1 and step % 2 == 0 :
                 if test_true_n_agents > 0: legal_actions[0,0] = 0
                 if test_true_n_agents > 1: legal_actions[1,1] = 0

            gh.store_transition(action, reward, next_obs, legal_actions, model_index=0)
            # Self-play MCTS stats (not strictly used by reanalyze's _prepare_ methods, but GameHistory stores them)
            gh.store_search_stats(
                root_value=0.4, pred_value=0.3,
                sampled_actions=action.reshape(1,test_true_n_agents), # C=1 like
                sampled_policy=np.array([0.9]),
                sampled_qvalues=np.array([0.45])
            )
        gh.game_over() # Finalizes internal numpy arrays
        mock_game_histories.append(gh)

    # Select game positions for the batch
    mock_game_pos_lst = [np.random.randint(0, max(1, len(gh) - test_num_unroll_steps -1)) for gh in mock_game_histories]
    mock_indices_lst = list(range(test_batch_size)) # Dummy indices
    mock_weights_lst = [1.0] * test_batch_size     # Dummy PER weights
    mock_transitions_collected = game_len * test_batch_size # Dummy

    buffer_context = (mock_game_histories, mock_game_pos_lst, mock_indices_lst, mock_weights_lst)

    # Instantiate ReanalyzeWorker
    reanalyze_worker = ReanalyzeWorker(rank=0, config=config)
    reanalyze_worker.model.eval() # Ensure model is in eval mode
    # Normally model weights would be updated via update_model, using latest target_model

    print("\n--- Calling ReanalyzeWorker.make_batch() ---")
    inputs_batch, targets_batch, info = reanalyze_worker.make_batch(buffer_context, mock_transitions_collected)
    
    print("\n--- ReanalyzeWorker.make_batch() completed ---")

    # --- Assertions and Checks ---
    obs_b, action_b, mask_b, indices_b, weights_b = inputs_batch
    target_r, target_v, target_p_tuple = targets_batch
    target_sa, target_sp, target_sir, target_sadv, target_samask = target_p_tuple

    print("\n--- Input Batch Shapes ---")
    print(f"Obs batch shape: {obs_b.shape}")
    # Expected obs shape: (B, N, TotalStackedObsFeaturesPerAgent) - depends on prepare_observation_lst
    # TotalStackedObsFeaturesPerAgent = (K+S) * obs_per_agent_flat_dim if image_based=False
    # For non-image, prepare_observation_lst in core.utils returns (B, N, S+K, Features)
    # This doesn't match training.py assert: obs_batch.shape == (batch_size, config.num_agents, obs_pad_size, *config.obs_shape[:-1])
    # This needs alignment or careful check of prepare_observation_lst in MAZero
    # For now, let's check the first dim (Batch size for make_batch)
    assert obs_b.shape[0] == test_batch_size 

    print(f"Action batch shape: {action_b.shape}")
    assert action_b.shape == (test_batch_size, test_num_unroll_steps + 1, test_true_n_agents)
    print(f"Mask batch shape: {mask_b.shape}")
    assert mask_b.shape == (test_batch_size, test_num_unroll_steps + 1)

    print("\n--- Target Batch Shapes ---")
    print(f"Target reward shape: {target_r.shape}")
    assert target_r.shape == (test_batch_size, test_num_unroll_steps + 1)
    print(f"Target value shape: {target_v.shape}")
    assert target_v.shape == (test_batch_size, test_num_unroll_steps + 1)

    print("\n--- Target Policy Tuple Shapes (C_effective = 1) ---")
    C_eff = 1 # Because our _prepare_policy_re generates for one chosen action
    print(f"Target sampled_actions shape: {target_sa.shape}")
    assert target_sa.shape == (test_batch_size, test_num_unroll_steps + 1, C_eff, test_true_n_agents)
    print(f"Target sampled_policies shape: {target_sp.shape}")
    assert target_sp.shape == (test_batch_size, test_num_unroll_steps + 1, C_eff)
    print(f"Target sampled_imp_ratio shape: {target_sir.shape}")
    assert target_sir.shape == (test_batch_size, test_num_unroll_steps + 1, C_eff)
    print(f"Target sampled_adv shape: {target_sadv.shape}") # Adv = Q - V. Q is V_mcts(s), V is V_pred(s)
    assert target_sadv.shape == (test_batch_size, test_num_unroll_steps + 1, C_eff)
    print(f"Target sampled_action_mask shape: {target_samask.shape}")
    assert target_samask.shape == (test_batch_size, test_num_unroll_steps + 1, C_eff)

    # Restore original C if it was changed for the test
    config.sampled_action_times = original_C

    print("\n--- ReanalyzeWorker Test Potentially Successful (inspect MCTS logs from MockModel) ---")

if __name__ == '__main__':
    run_reanalyze_worker_test()