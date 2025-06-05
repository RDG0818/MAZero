# unit_test_data_worker.py

import logging
import math
import os
import time
from typing import Tuple, List, Union, Any, NamedTuple
from abc import ABC, abstractmethod
from core.config import BaseConfig
from gymnasium.utils import seeding


import numpy as np
import torch
import torch.nn as nn
# Ensure these are importable from your project structure or place mocks here
# from core.config import BaseConfig # We'll mock this
# from core.model import BaseNet, NetworkOutput # We'll mock these
# from core.game import Game, GameHistory # We'll mock Game, use real GameHistory
# from core.replay_buffer import ReplayBuffer # We'll mock this
# from core.storage import SharedStorage # We'll mock this

# Assuming your modified files are in the same directory or accessible via PYTHONPATH
from mcts_sampled_modified import SampledMCTS, SearchOutput
from selfplay_worker_modified import DataWorker
from core.game import GameHistory # Using the real GameHistory

# --- Start Mock/Minimal Definitions ---

class MockAction(NamedTuple): # If Action is a NamedTuple in your core.model
    action: torch.Tensor

class MockHiddenState(torch.Tensor): # If HiddenState is just a Tensor
    pass

class NetworkOutput(NamedTuple): # From core.model
    value: Any
    reward: Any
    policy_logits: Any
    hidden_state: Any
    # Add if your model predicts this and SampledMCTS uses it:
    # legal_actions_logits: Any = None

class BaseNet(nn.Module, ABC): # Minimal BaseNet for MockModel
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

class Game(ABC): # Minimal Game for MockEnvGame
    def __init__(self, env_config=None, seed=None): # Added env_config and seed
        self.n_agents = 0
        self.obs_size = 0
        self.action_space_size = 0
    @abstractmethod
    def legal_actions(self): pass
    @abstractmethod
    def get_max_episode_steps(self) -> int: pass
    @abstractmethod
    def step(self, action: List[int]) -> Tuple[np.ndarray, float, bool, dict]: pass
    @abstractmethod
    def reset(self) -> np.ndarray: pass
    def close(self, *args, **kwargs): pass
    def render(self, *args, **kwargs): pass


class MockConfig(BaseConfig if 'BaseConfig' in globals() else object): # Inherit if BaseConfig is defined
    def __init__(self, true_n_agents, action_space_sz, hidden_sz_per_agent, num_sims, num_pmcts_val=1, obs_feat_size=10):
        self.selfplay_on_gpu = False # Force CPU for easier testing
        self.num_unroll_steps = 2 # Small for testing
        self.td_steps = 3         # Small for testing
        self.action_space_size = action_space_sz
        self.num_agents = true_n_agents # True number of agents
        self.case = 'test_case'
        self.stacked_observations = 1 # Keep simple
        self.seed = 42
        self.image_based = False
        self.max_moves = 5 # Short episodes for testing
        self.num_pmcts = num_pmcts_val # Number of parallel environments for the worker

        # For GameHistory
        self.discount = 0.99
        self.cvt_string = False
        self.gray_scale = False
        self.sampled_action_times = 3 # K for Sampled MCTS in GameHistory padding

        # For MCTS (from SampledMCTS test)
        self.pb_c_base = 19652.0
        self.pb_c_init = 1.25
        self.mcts_rho = 0.75
        self.mcts_lambda = 0.8
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25
        self.num_simulations = num_sims
        self.tree_value_stat_delta_lb = 0.01
        
        # For Model (used by MockModel constructor)
        self.hidden_state_size_per_agent = hidden_sz_per_agent 
        self.obs_size_per_agent = obs_feat_size 

        # For PER
        self.use_priority = True
        self.use_max_priority = False
        self.prioritized_replay_eps = 1e-6

        # For visit_softmax_temperature_fn and eps_greedy_fn
        self.training_steps = 1000 # Example total training steps for temp scheduling
        self.eps_start = 0.0 # No explicit eps-greedy beyond MCTS for this test
        self.eps_end = 0.0
        self.eps_annealing_time = 1

    def get_uniform_network(self): # Method expected by DataWorker
        return MockModel(self, torch.device("cpu")) # Pass self as config

    def new_game(self, seed=None): # Method expected by DataWorker
        return MockEnvGame(self, seed=seed) # Pass self as config

    def visit_softmax_temperature_fn(self, trained_steps): return 1.0
    def sampled_action_temperature_fn(self, trained_steps): return 1.0
    def eps_greedy_fn(self, trained_steps): return 0.0 # Disable extra epsilon greedy


class MockModel(BaseNet):
    def __init__(self, config: MockConfig, device: torch.device):
        super().__init__()
        self.true_num_agents = config.num_agents
        self.action_space_size = config.action_space_size
        self.hidden_size_per_agent = config.hidden_state_size_per_agent
        self.total_hidden_size = self.true_num_agents * self.hidden_size_per_agent
        self.device = device
        self.dummy_param = torch.nn.Parameter(torch.empty(0))
        print(f"[MockModel Init] true_num_agents: {self.true_num_agents}, action_space_size: {self.action_space_size}, total_hidden_size: {self.total_hidden_size}")

    def initial_inference(self, observation: torch.Tensor) -> NetworkOutput:
        batch_size = observation.shape[0] # observation is [B_active, Stack, N, Obs_per_agent] or [B_active, N*Obs_per_agent]
                                          # if prepare_observation_lst flattens N. Let's assume it's shaped for the model.
                                          # For this test, we'll assume obs is already [B_active, N*Obs_per_agent] if model expects that.
        print(f"\n[MockModel.initial_inference] Input obs batch_size: {batch_size}")
        hidden_state = torch.rand((batch_size, self.total_hidden_size), device=self.device) * 0.1
        reward_np = np.zeros((batch_size, 1), dtype=np.float32)
        value_np = np.full((batch_size, 1), 0.5, dtype=np.float32)
        policy_logits_np = np.zeros((batch_size, self.true_num_agents, self.action_space_size), dtype=np.float32)
        # Make agent 0 prefer action 0, agent 1 prefer action 1 etc. for predictability
        for i in range(min(self.true_num_agents, self.action_space_size)):
             policy_logits_np[:, i, i] = 1.0 
        
        return NetworkOutput(value=value_np, reward=reward_np, policy_logits=policy_logits_np, hidden_state=hidden_state)

    def recurrent_inference(self, hidden_state: torch.Tensor, action: torch.Tensor) -> NetworkOutput:
        batch_size = hidden_state.shape[0]
        print(f"\n[MockModel.recurrent_inference] Called with:")
        print(f"  hidden_state shape: {hidden_state.shape}")
        print(f"  action shape: {action.shape}, action values:\n{action.cpu().numpy()}")
        assert hidden_state.shape[1] == self.total_hidden_size
        assert action.shape[1] == self.true_num_agents

        next_hidden_state = hidden_state + 0.01 * torch.randn_like(hidden_state) # Add some noise
        reward_np = np.ones((batch_size, 1), dtype=np.float32) * (0.1 + np.sum(action.cpu().numpy(), axis=1, keepdims=True)*0.01) # Reward depends on sum of actions
        value_np = np.full((batch_size, 1), 0.6, dtype=np.float32)
        policy_logits_np = np.random.rand(batch_size, self.true_num_agents, self.action_space_size).astype(np.float32) * 0.1
        for i in range(min(self.true_num_agents, self.action_space_size)):
             policy_logits_np[:, i, (i+1)%self.action_space_size] = 1.0 # Prefer next action

        return NetworkOutput(value=value_np, reward=reward_np, policy_logits=policy_logits_np, hidden_state=next_hidden_state)
    
    def representation(self, observation: torch.Tensor) -> torch.Tensor: return self.initial_inference(observation).hidden_state
    def prediction(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # This is called by SampledMCTS if initial_inference is too heavy for internal nodes
        # For MAMuZeroNet, initial_inference calls representation and prediction.
        # Let's make this consistent.
        policy_logits, value_logits = torch.zeros(1), torch.zeros(1) # Placeholder, should align with prediction_network output
        return policy_logits, value_logits
    def dynamics(self, hidden_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: return torch.empty(0), torch.empty(0)


class MockEnvGame(Game):
    def __init__(self, config: MockConfig, seed=None):
        super().__init__()
        self.n_agents = config.num_agents
        self.obs_size = config.obs_size_per_agent # Assuming obs_size in Game refers to per-agent
        self.action_space_size = config.action_space_size
        self._episode_steps = 0
        self._max_steps = config.max_moves
        self.np_random, _ = seeding.np_random(seed)
        print(f"[MockEnvGame Init] n_agents: {self.n_agents}, obs_size: {self.obs_size}, action_space: {self.action_space_size}, max_steps: {self._max_steps}")


    def legal_actions(self) -> List[List[int]]:
        # All actions legal for all agents for simplicity in this mock
        # unless an agent is "dead" (conceptual, not implemented here for simplicity)
        return np.ones((self.n_agents, self.action_space_size), dtype=np.int32).tolist()

    def get_max_episode_steps(self) -> int:
        return self._max_steps

    def step(self, action: List[int]) -> Tuple[np.ndarray, float, bool, dict]:
        self._episode_steps += 1
        print(f"[MockEnvGame.step] Received joint action: {action} (Step: {self._episode_steps})")
        assert len(action) == self.n_agents, f"Action length mismatch. Expected {self.n_agents}, got {len(action)}"
        
        # Mock observation per agent
        next_obs_per_agent = [self.np_random.random(self.obs_size).astype(np.float32) for _ in range(self.n_agents)]
        next_obs_stacked = np.array(next_obs_per_agent) # Shape: (N, ObsSize)
        
        reward = 0.1 * self._episode_steps # Simple reward
        done = self._episode_steps >= self._max_steps
        info = {'battle_won': True if done and reward > 0.3 else False} # Example info

        return next_obs_stacked, reward, done, info

    def reset(self) -> np.ndarray:
        print("[MockEnvGame.reset] Called.")
        self._episode_steps = 0
        obs_per_agent = [self.np_random.random(self.obs_size).astype(np.float32) for _ in range(self.n_agents)]
        return np.array(obs_per_agent) # Shape: (N, ObsSize)

class MockReplayBuffer:
    def __init__(self, config=None):
        self.saved_pools_count = 0
        print("[MockReplayBuffer Init]")
    def save_pools(self, trajectory_pool: List[Tuple[GameHistory, Any]]):
        self.saved_pools_count += 1
        print(f"[MockReplayBuffer.save_pools] Called. Pool size: {len(trajectory_pool)}. Trajectory lengths: {[len(gh) for gh, _ in trajectory_pool]}")
    def sample_batch(self, batch_size: int, beta: float = 0.0): return None # For ReanalyzeWorker
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray): pass # For ReanalyzeWorker

class MockSharedStorage:
    def __init__(self, config=None):
        self._trained_steps = 0
        self._start_signal = True # Assume training can start
        self._weights = (0, None) # model_idx, weights_dict
        print("[MockSharedStorage Init]")
    def add_worker_logs(self, log_dict: dict):
        print(f"[MockSharedStorage.add_worker_logs] Received log: {log_dict}")
    def get_counter(self): return self._trained_steps
    def get_start_signal(self): return self._start_signal
    def get_weights(self): return self._weights
    def get_target_weights(self): return self._weights # For reanalyze
    def set_weights(self, model_idx, weights): self._weights = (model_idx, weights)
# --- End Mock Definitions ---


def run_data_worker_test():
    print("--- Starting DataWorker Test ---")
    # Test Parameters
    test_true_n_agents = 2
    test_action_space_sz = 3
    test_hidden_sz_per_agent = 8 
    test_num_sims = 3 # Low for speed
    test_num_pmcts = 1 # Number of parallel environments for the worker
    test_obs_feat_size = 5

    # Setup Config
    config = MockConfig(
        true_n_agents=test_true_n_agents,
        action_space_sz=test_action_space_sz,
        hidden_sz_per_agent=test_hidden_sz_per_agent,
        num_sims=test_num_sims,
        num_pmcts_val=test_num_pmcts,
        obs_feat_size=test_obs_feat_size
        
    )
    config.max_moves = 3 # Very short episodes

    # Setup Mocks
    mock_replay_buffer = MockReplayBuffer(config)
    mock_shared_storage = MockSharedStorage(config)
    
    # Instantiate DataWorker
    # Note: DataWorker's __init__ calls self.init_envs() which calls config.new_game() and env.reset()
    # It also calls config.get_uniform_network()
    data_worker = DataWorker(
        rank=0,
        config=config,
        replay_buffer=mock_replay_buffer,
        shared_storage=mock_shared_storage
    )
    data_worker.model.eval() # Ensure model is in eval mode

    # Call the run method
    print("\n--- Calling DataWorker.run() ---")
    transitions = data_worker.run(start_training=True, trained_steps=0)
    print(f"\n--- DataWorker.run() completed. Transitions collected: {transitions} ---")

    # --- Assertions and Checks ---
    assert transitions > 0, "No transitions were collected."
    assert mock_replay_buffer.saved_pools_count > 0, "Replay buffer save_pools was not called."
    
    # Check GameHistory content for the first environment (if num_pmcts=1)
    if config.num_pmcts == 1 and data_worker.trajectory_pool: # trajectory_pool might be cleared
        # To get the last game history, we'd need to intercept it or have mock_replay_buffer store it
        # For now, let's assume it ran at least one episode
        print("Test assumes one episode was run and trajectory_pool might be empty due to _free.")
        # A more robust test would involve a ReplayBuffer mock that allows inspecting saved data.
    
    print("\n--- DataWorker Test Potentially Successful (inspect logs for details) ---")

if __name__ == '__main__':
    run_data_worker_test()