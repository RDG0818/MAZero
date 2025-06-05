from typing import List, Tuple, Any, NamedTuple
from abc import ABC, abstractmethod # Added for BaseNet

import numpy as np
import torch
import torch.nn as nn # Added for BaseNet
# Make sure 'mcts_sampled_modified.py' contains your modified SampledMCTS class
# and is in the same directory or your PYTHONPATH.
from mcts_sampled_modified import SampledMCTS, SearchOutput # Assuming your file is named this

# ---- Minimal core.model and core.config structures for the test ----
class HiddenState(torch.Tensor): pass # Dummy for type hint if needed (not strictly necessary)
class Action(torch.Tensor): pass    # Dummy

class NetworkOutput(NamedTuple):
    value: Any
    reward: Any
    policy_logits: Any
    hidden_state: Any
    # Add if your model will predict this:
    # legal_actions_logits: Any = None

class BaseNet(nn.Module, ABC): # Made it inherit from nn.Module and ABC
    def __init__(self, inverse_value_transform=None, inverse_reward_transform=None): # Default args
        super().__init__()
        self._inverse_value_transform = inverse_value_transform
        self._inverse_reward_transform = inverse_reward_transform
    @abstractmethod
    def representation(self, observation: torch.Tensor) -> torch.Tensor: pass
    @abstractmethod
    def prediction(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: pass # Original was missing legal_actions from tuple
    @abstractmethod
    def dynamics(self, hidden_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: pass
    @abstractmethod
    def initial_inference(self, observation: torch.Tensor) -> NetworkOutput: pass
    @abstractmethod
    def recurrent_inference(self, hidden_state: torch.Tensor, action: torch.Tensor) -> NetworkOutput: pass
    def get_weights(self): return {k: v.cpu() for k, v in self.state_dict().items()}
    def set_weights(self, weights): self.load_state_dict(weights)

class MockConfig: # Simplified from BaseConfig for testing
    def __init__(self, true_n_agents_for_test, action_space_sz_for_test, hidden_sz_per_agent_for_test, num_sims_for_test):
        # MCTS & UCB parameters
        self.pb_c_base = 19652.0
        self.pb_c_init = 1.25
        self.discount = 0.99
        self.mcts_rho = 0.75
        self.mcts_lambda = 0.8
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25
        self.num_simulations = num_sims_for_test
        self.action_space_size = action_space_sz_for_test # Per agent
        self.num_agents = true_n_agents_for_test # True number of agents in env for this config instance
        self.sampled_action_times = 5
        self.tree_value_stat_delta_lb = 0.01
        self.hidden_state_size_per_agent = hidden_sz_per_agent_for_test
        # For model
        self.fc_representation_layers = [64] # Example
        self.fc_dynamic_layers = [64]
        self.fc_reward_layers = [32]
        self.fc_value_layers = [32]
        self.fc_policy_layers = [32]
        self.reward_support_size = 1 # Assuming scalar reward for simplicity in mock
        self.value_support_size = 1  # Assuming scalar value for simplicity in mock
        self.obs_size_per_agent = 10 # Example observation feature size per agent

# ---- Mock Model ----
class MockModel(BaseNet):
    def __init__(self, config: MockConfig, device: torch.device):
        super().__init__()
        self.true_num_agents = config.num_agents
        self.action_space_size = config.action_space_size
        self.hidden_size_per_agent = config.hidden_state_size_per_agent
        self.total_hidden_size = self.true_num_agents * self.hidden_size_per_agent
        self.device = device
        self.dummy_param = torch.nn.Parameter(torch.empty(0)) # To make it an nn.Module

    def initial_inference(self, observation: torch.Tensor) -> NetworkOutput:
        batch_size = observation.shape[0]
        print(f"\n[MockModel.initial_inference] Obs shape: {observation.shape}")
        
        # For this test, observation content might not matter as much as shapes.
        # Hidden state should be for all agents combined if that's what MCTS root expects.
        hidden_state = torch.rand((batch_size, self.total_hidden_size), device=self.device) * 0.1
        reward_np = np.zeros((batch_size, 1), dtype=np.float32) # scalar reward
        value_np = np.full((batch_size, 1), 0.5, dtype=np.float32) # Predictable value
        # Policy logits for ALL agents at this state
        policy_logits_np = np.zeros((batch_size, self.true_num_agents, self.action_space_size), dtype=np.float32)
        
        print(f"[MockModel.initial_inference] Outputting: hidden_state_shape: {hidden_state.shape}, "
              f"policy_logits_shape: {policy_logits_np.shape}")

        return NetworkOutput(
            value=value_np,
            reward=reward_np,
            policy_logits=policy_logits_np, # For MCTS, needs to be numpy if add_noise=True in batch_search
            hidden_state=hidden_state      # This is torch.Tensor for model's internal use
        )

    def recurrent_inference(self, hidden_state: torch.Tensor, action: torch.Tensor) -> NetworkOutput:
        batch_size = hidden_state.shape[0]
        print(f"\n[MockModel.recurrent_inference] Called with:")
        print(f"  hidden_state shape: {hidden_state.shape}")
        print(f"  action shape: {action.shape}, action values:\n{action.cpu().numpy()}") # Print action values
        
        assert hidden_state.shape == (batch_size, self.total_hidden_size), \
            f"Expected hs shape {(batch_size, self.total_hidden_size)}, got {hidden_state.shape}"
        assert action.shape == (batch_size, self.true_num_agents), \
            f"Expected action shape {(batch_size, self.true_num_agents)}, got {action.shape}"

        next_hidden_state = hidden_state + 0.01 # Simple deterministic change, keep on device
        reward_np = np.ones((batch_size, 1), dtype=np.float32) * 0.1 # Predictable reward
        value_np = np.full((batch_size, 1), 0.6, dtype=np.float32) # Predictable next value
        # Policy logits for ALL agents at the NEXT state
        policy_logits_np = np.zeros((batch_size, self.true_num_agents, self.action_space_size), dtype=np.float32)
        
        print(f"[MockModel.recurrent_inference] Outputting: next_hidden_state_shape: {next_hidden_state.shape}, "
              f"policy_logits_shape: {policy_logits_np.shape}")

        return NetworkOutput(
            value=value_np,
            reward=reward_np,
            policy_logits=policy_logits_np,
            hidden_state=next_hidden_state
        )

    # Dummy implementations for abstract methods not critical to this specific test
    def representation(self, observation: torch.Tensor) -> torch.Tensor: return torch.empty(0)
    def prediction(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: return torch.empty(0), torch.empty(0)
    def dynamics(self, hidden_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: return torch.empty(0), torch.empty(0)

# ---- Test Script ----
def run_sequential_mcts_test():
    # Test Parameters
    test_batch_size = 1
    test_true_n_agents = 2
    test_action_space_sz = 3
    test_hidden_sz_per_agent = 4
    test_num_sims = 3 # Keep low for quick test, increase for more MCTS activity

    device = torch.device("cpu")

    # Setup Config
    config = MockConfig(test_true_n_agents, test_action_space_sz, test_hidden_sz_per_agent, test_num_sims)

    # Setup Model
    mock_model = MockModel(config, device).to(device)
    mock_model.eval()

    # Setup MCTS object
    mcts_instance = SampledMCTS(config, np_random=np.random.RandomState(123)) # Seeded for reproducibility

    # Prepare initial mock observation for the model
    # Shape: (batch_size, true_num_agents, obs_size_per_agent)
    mock_observation_np = np.random.rand(test_batch_size, test_true_n_agents, config.obs_size_per_agent)
    mock_observation_torch = torch.from_numpy(mock_observation_np).float().to(device)
    
    # This is NetworkOutput for the root state s0.
    # Its policy_logits field is for ALL agents at s0.
    initial_net_out_s0 = mock_model.initial_inference(mock_observation_torch)
    print("\nInitial Network Output for s0 (passed to first batch_search):")
    print(f"  Value: {initial_net_out_s0.value}, Reward: {initial_net_out_s0.reward}")
    print(f"  Policy Logits shape: {initial_net_out_s0.policy_logits.shape}")
    print(f"  Hidden State shape: {initial_net_out_s0.hidden_state.shape}")


    # Legal actions for the root state (e.g., all actions legal for all agents for this test)
    # Shape: (batch_size, true_num_agents, action_space_size)
    legal_actions_root = np.ones((test_batch_size, test_true_n_agents, config.action_space_size), dtype=np.int32)
    # Example: Make action 0 illegal for agent 1 at root to test masking for current_agent_idx
    if test_true_n_agents > 1 :
         legal_actions_root[0, 1, 0] = 0


    # --- Sequential MCTS Loop (mimicking what selfplay_worker would do) ---
    chosen_actions_for_step_np = np.full((test_batch_size, test_true_n_agents), -1, dtype=np.int32)
    
    for agent_idx_turn in range(test_true_n_agents):
        print(f"\n{'='*20} Running batch_search for agent_idx: {agent_idx_turn} {'='*20}")
        
        fixed_previous_actions_for_agent_np = None
        if agent_idx_turn > 0:
            fixed_previous_actions_for_agent_np = chosen_actions_for_step_np[:, :agent_idx_turn].copy()
        
        print(f"Input 'factor' (fixed_previous_actions): {fixed_previous_actions_for_agent_np}")
        print(f"Input 'legal_actions_lst' (for root node): shape {legal_actions_root.shape}")

        # The initial_net_out_s0 provides the starting hidden_state for MCTS,
        # and the policy_logits from it will be sliced for current_agent_idx's priors.
        search_output_agent_k = mcts_instance.batch_search(
            model=mock_model,
            network_output=initial_net_out_s0, # Root state predictions for all agents
            current_agent_idx=agent_idx_turn,
            factor=fixed_previous_actions_for_agent_np,
            true_num_agents=test_true_n_agents,
            legal_actions_lst=legal_actions_root, # True legal actions at the root for all agents
            device=device,
            add_noise=True 
        )

        print(f"\n--- SearchOutput for agent {agent_idx_turn} ---")
        print(f"  Value (shape {search_output_agent_k.value.shape}): {search_output_agent_k.value}")
        assert search_output_agent_k.value.shape == (test_batch_size,)
        
        print(f"  Marginal visit count (shape {search_output_agent_k.marginal_visit_count.shape}): \n{search_output_agent_k.marginal_visit_count}")
        assert search_output_agent_k.marginal_visit_count.shape == (test_batch_size, 1, config.action_space_size)

        if search_output_agent_k.sampled_actions:
            print(f"  Sampled actions for agent {agent_idx_turn} (MCTS root children actions, shape {search_output_agent_k.sampled_actions[0].shape}): \n{search_output_agent_k.sampled_actions[0]}")
            assert search_output_agent_k.sampled_actions[0].shape[1] == 1 # agent_num=1 for cytree

            # Simulate action selection for this agent based on MCTS result (e.g., from visit counts)
            # For batch_size = 1:
            visit_counts_agent_k = search_output_agent_k.marginal_visit_count[0, 0, :]
            if np.sum(visit_counts_agent_k) > 0:
                policy_agent_k = visit_counts_agent_k / np.sum(visit_counts_agent_k)
                # chosen_action_for_agent_k = mcts_instance.np_random.choice(config.action_space_size, p=policy_agent_k)
                chosen_action_for_agent_k = np.argmax(policy_agent_k) # Deterministic for test
            else: # Should not happen with noise if legal actions exist
                print(f"WARNING: Agent {agent_idx_turn} has zero visit counts for all actions!")
                chosen_action_for_agent_k = mcts_instance.np_random.choice(config.action_space_size)


            print(f"  Agent {agent_idx_turn} chosen action: {chosen_action_for_agent_k}")
            chosen_actions_for_step_np[0, agent_idx_turn] = chosen_action_for_agent_k
        else:
            print(f"  WARNING: Agent {agent_idx_turn} NO sampled actions from MCTS!")
            chosen_actions_for_step_np[0, agent_idx_turn] = 0 # Fallback, should investigate

    print(f"\n{'='*20} Final chosen joint action for the step: {chosen_actions_for_step_np} {'='*20}")

if __name__ == '__main__':
    # This ensures the script can be run directly.
    # Your modified SampledMCTS class needs to be accessible.
    # Either paste it above, or ensure mcts_sampled_modified.py is in the same directory
    # or your PYTHONPATH.
    run_sequential_mcts_test()