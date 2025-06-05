import logging
import math
import os
import time
from typing import Tuple, List, Union

import numpy as np
import ray
from ray.actor import ActorHandle
import torch
from torch.cuda.amp import autocast as autocast
from gymnasium.utils import seeding

from core.mcts.tree_search.mcts_sampled import SearchOutput, SampledMCTS
from core.config import BaseConfig
from core.replay_buffer import ReplayBuffer
from core.storage import SharedStorage
from core.game import GameHistory
from core.utils import select_action, prepare_observation_lst, get_max_entropy, eps_greedy_action


class DataWorker(object):
    def __init__(self, rank, config: BaseConfig, replay_buffer: ReplayBuffer, shared_storage: SharedStorage):
        """Data Worker for collecting data through self-play

        Parameters
        ----------
        rank: int
            id of the worker
        replay_buffer: Any
            Replay buffer to save self-play data
        shared_storage: Any
            The share storage to control & get latest model
        """
        self.rank = rank
        self.config = config
        self.replay_buffer = replay_buffer
        self.shared_storage = shared_storage
        self.np_random, _ = seeding.np_random(config.seed * 1000 + self.rank)

        self.device = 'cuda' if (config.selfplay_on_gpu and torch.cuda.is_available()) else 'cpu'
        self.gap_step = self.config.num_unroll_steps + self.config.td_steps
        self.max_visit_entropy = get_max_entropy(self.config.action_space_size)

        # create env & logs
        self.init_envs()

        # create model from game_config
        self.model = self.config.get_uniform_network()
        self.model.to(self.device)
        self.model.eval()
        self.last_model_index = -1

        self.trajectory_pool = []
        self.pool_size = 1  # max size for buffering pool

    def init_envs(self):
        num_envs = self.config.num_pmcts

        self.ray_store_obs = False

        self.envs = [
            self.config.new_game(self.config.seed + (self.rank + 1) * i)
            for i in range(num_envs)
        ]

        self.eps_steps_lst = np.zeros(num_envs)
        self.eps_reward_lst = np.zeros(num_envs)
        self.visit_entropies_lst = np.zeros(num_envs)
        self.model_index_lst = np.zeros(num_envs)
        if self.config.case in ['smac', 'gfootball']:
            self.battle_won_lst = np.zeros(num_envs)

        init_obses = [env.reset() for env in self.envs]
        self.game_histories = [None for _ in range(num_envs)]       # type: list[GameHistory]

        # stack observation windows in boundary
        self.stack_obs_windows = [[] for _ in range(num_envs)]
        # initial stack observation: [s0, s0, s0, s0]
        for i in range(num_envs):
            self.stack_obs_windows[i] = [init_obses[i] for _ in range(self.config.stacked_observations)]
            self.game_histories[i] = GameHistory(config=self.config, ray_store_obs=self.ray_store_obs)
            self.game_histories[i].init(self.stack_obs_windows[i])

        # for priorities in self-play
        self.pred_values_lst = [[] for _ in range(num_envs)]    # pred value
        self.search_values_lst = [[] for _ in range(num_envs)]  # target value

        self.dones = np.zeros(num_envs, dtype=np.bool_)

    def put(self, data: Tuple[GameHistory, List[float]]):
        # put a game history into the pool
        self.trajectory_pool.append(data)

    def _free(self):
        # save the game histories and clear the pool
        if len(self.trajectory_pool) >= self.pool_size:
            self.replay_buffer.save_pools(self.trajectory_pool)
            del self.trajectory_pool[:]

    def _log_to_buffer(self, log_dict: dict):
        self.shared_storage.add_worker_logs(log_dict)

    def _update_model_before_step(self):
        # no update when serial
        return

    def update_model(self, model_index, weights):
        self.model.set_weights(weights)
        self.last_model_index = model_index

    def log(self, env_id, **kwargs):
        # send logs
        log_dict = {
            'eps_len': self.eps_steps_lst[env_id],
            'eps_reward': self.eps_reward_lst[env_id],
            'visit_entropy': self.visit_entropies_lst[env_id] / max(self.eps_steps_lst[env_id], 1),
            'model_index': self.model_index_lst[env_id] / max(self.eps_steps_lst[env_id], 1),
        }
        for k, v in kwargs.items():
            log_dict[k] = v
        if self.config.case in ['smac', 'gfootball']:
            log_dict['win_rate'] = self.battle_won_lst[env_id]

        self._log_to_buffer(log_dict)

    def reset_env(self, env_id):
        self.eps_steps_lst[env_id] = 0
        self.eps_reward_lst[env_id] = 0
        self.visit_entropies_lst[env_id] = 0
        self.model_index_lst[env_id] = 0
        if self.config.case in ['smac', 'gfootball']:
            self.battle_won_lst[env_id] = 0
        # new trajectory
        init_obs = self.envs[env_id].reset()
        self.stack_obs_windows[env_id] = [init_obs for _ in range(self.config.stacked_observations)]
        self.game_histories[env_id] = GameHistory(config=self.config, ray_store_obs=self.ray_store_obs)
        self.game_histories[env_id].init(self.stack_obs_windows[env_id])
        self.pred_values_lst[env_id] = []
        self.search_values_lst[env_id] = []
        self.dones[env_id] = False

    def get_priorities(self, pred_values: List[float], search_values: List[float]) -> Union[List[float], None]:
        # obtain the priorities
        if self.config.use_priority and not self.config.use_max_priority:
            priorities = np.abs(np.asarray(pred_values) - np.asarray(search_values)) + self.config.prioritized_replay_eps
        else:
            # priorities is None -> use the max priority for all newly collected data
            priorities = None

        return priorities

    def run(self, start_training: bool = False, trained_steps: int = 0) -> int:
        num_envs = self.config.num_pmcts
        episodes_collected = 0
        transitions_collected = 0
        true_num_agents = self.config.num_agents
        agent_actions = np.full((num_envs, true_num_agents), -1, dtype=np.int32)
        agent_search_outputs = [[None for _ in range(true_num_agents)] for _ in range(num_envs)]
        entropies = [[0.0 for _ in range(true_num_agents)] for _ in range(num_envs)]

        with torch.no_grad():
            # play games until max episodes
            while episodes_collected < num_envs:

                # set temperature for distributions
                temperature = self.config.visit_softmax_temperature_fn(trained_steps)
                sampled_tau = self.config.sampled_action_temperature_fn(trained_steps)
                greedy_epsilon = self.config.eps_greedy_fn(trained_steps)

                active_env_indices = [i for i in range(num_envs) if not self.dones[i]]

                if not active_env_indices: # All envs might be done, break to avoid issues
                    break

                # update model
                self._update_model_before_step()

                # stack obs for model inference
                stack_obs = [self.game_histories[i].step_obs() for i in active_env_indices]
                stack_obs = prepare_observation_lst(stack_obs, self.config.image_based)
                if self.config.image_based:
                    stack_obs = torch.from_numpy(stack_obs).to(self.device).float() / 255.0
                else:
                    stack_obs = torch.from_numpy(stack_obs).to(self.device).float()

                with autocast():
                    network_output = self.model.initial_inference(stack_obs)
                legal_actions_lst = np.asarray([self.envs[i].legal_actions() for i in active_env_indices]) # Shape: (num_envs, num_agents, action_space_size)

                mcts_handler = SampledMCTS(self.config, self.np_random)

                temp_agent_actions = np.full((len(active_env_indices), true_num_agents), -1, dtype=np.int32)
                temp_entropies = np.zeros((len(active_env_indices), true_num_agents))

                for agent_idx in range(true_num_agents):
                    factor = None
                    if agent_idx > 0:
                        factor = temp_agent_actions[:, :agent_idx].copy()
                    
                    agent_k_output = mcts_handler.batch_search(
                        model=self.model,
                        network_output=network_output,
                        current_agent_idx=agent_idx,
                        factor=factor,
                        true_num_agents=true_num_agents,
                        legal_actions_lst=legal_actions_lst,
                        device=self.device,
                        add_noise=True,
                        sampled_tau=sampled_tau
                    )

                    for active_env_local_idx, original_env_idx in enumerate(active_env_indices):
                        # Since agent_k_output is batched, we slice for each env
                        agent_search_outputs[original_env_idx][agent_idx] = SearchOutput(
                            value=agent_k_output.value[active_env_local_idx:active_env_local_idx+1],
                            marginal_visit_count=agent_k_output.marginal_visit_count[active_env_local_idx:active_env_local_idx+1],
                            marginal_priors=agent_k_output.marginal_priors[active_env_local_idx:active_env_local_idx+1],
                            sampled_actions=[agent_k_output.sampled_actions[active_env_local_idx]],
                            sampled_visit_count=[agent_k_output.sampled_visit_count[active_env_local_idx]],
                            sampled_pred_probs=[agent_k_output.sampled_pred_probs[active_env_local_idx]],
                            sampled_beta=[agent_k_output.sampled_beta[active_env_local_idx]],
                            sampled_beta_hat=[agent_k_output.sampled_beta_hat[active_env_local_idx]],
                            sampled_priors=[agent_k_output.sampled_priors[active_env_local_idx]],
                            sampled_imp_ratio=[agent_k_output.sampled_imp_ratio[active_env_local_idx]],
                            sampled_pred_values=[agent_k_output.sampled_pred_values[active_env_local_idx]],
                            sampled_mcts_values=[agent_k_output.sampled_mcts_values[active_env_local_idx]],
                            sampled_rewards=[agent_k_output.sampled_rewards[active_env_local_idx]],
                            sampled_qvalues=[agent_k_output.sampled_qvalues[active_env_local_idx]]
                        )

                        sampled_actions = agent_k_output.sampled_actions[active_env_local_idx]
                        sampled_visit_counts = agent_k_output.sampled_visit_count[active_env_local_idx]
                        single_agent_legal_actions = legal_actions_lst[active_env_local_idx, agent_idx, :]

                        if not sampled_actions.size:
                            legal_indices = np.where(single_agent_legal_actions == 1)[0]
                            agent_action = self.np_random.choice(legal_indices) if legal_indices.size > 0 else 0
                            visit_entropy_per_agent = 0.0
                        else:
                            action_pos, visit_entropy_per_agent = select_action(
                                sampled_visit_counts,
                                temperature=temperature,
                                deterministic=False,
                                np_random=self.np_random
                            )
                            agent_action = sampled_actions[action_pos, 0]
                        
                        # Epsilon-greedy
                        agent_action = eps_greedy_action(
                            agent_action,
                            single_agent_legal_actions,
                            greedy_epsilon
                        )

                        temp_agent_actions[active_env_local_idx, agent_idx] = agent_action
                        temp_entropies[active_env_local_idx][agent_idx] = visit_entropy_per_agent

                for active_env_local_idx, original_env_idx in enumerate(active_env_indices):
                    agent_actions[original_env_idx, :] = temp_agent_actions[active_env_local_idx, :]
                    entropies[original_env_idx] = np.mean(temp_entropies[active_env_local_idx, :]) if true_num_agents > 0 else 0
                        
                for i in active_env_indices:
                    
                    action = agent_actions[i, :]

                    next_obs, reward, done, info = self.envs[i].step(action)
                    self.dones[i] = done

                    # store data
                    self.game_histories[i].store_transition(action, reward, next_obs, legal_actions_lst[i], self.last_model_index)
                    
                    root_value = agent_search_outputs[i][0].value[0]
                    pred_value = network_output.value[active_env_indices.index(i)].item() # fix this apparently
                    sampled_actions = action.reshape(1, true_num_agents)

                    prob_action = 1.0
                    agent_entropies = []
                    for ag_idx in range(true_num_agents):
                        ag_action = action[ag_idx]
                        ag_so = agent_search_outputs[i][ag_idx]
                        ag_visits = ag_so.marginal_visit_count[0, 0, :]
                        curr_ag_entropy = 0.0
                        if np.sum(ag_visits) > 0:
                            prob_action_dist = ag_visits / np.sum(ag_visits)
                            prob_action *= prob_action_dist[ag_action]
                            curr_ag_entropy = -np.sum(prob_action_dist * np.log(prob_action_dist + 1e-9))
                        elif self.config.action_space_size > 0:
                            prob_action *= (1.0/self.config.action_space_size)
                        agent_entropies.append(curr_ag_entropy)

                    sampled_policy = np.array([prob_action])
                    if agent_entropies: self.visit_entropies_lst[i] += np.mean(agent_entropies)

                    sampled_qvalues = np.array([root_value])

                    self.game_histories[i].store_search_stats(root_value, pred_value, sampled_actions, sampled_policy, sampled_qvalues) 
                    if self.config.use_priority:
                        self.pred_values_lst[i].append(pred_value)
                        self.search_values_lst[i].append(root_value)

                    # update logs
                    self.eps_steps_lst[i] += 1
                    self.eps_reward_lst[i] += reward
                    self.model_index_lst[i] += self.last_model_index
                    if self.config.case in ['smac', 'gfootball']:
                        self.battle_won_lst[i] = info['battle_won']

                    # fresh stack windows
                    del self.stack_obs_windows[i][0]
                    self.stack_obs_windows[i].append(next_obs)

                    # if is the end of the game:
                    if self.dones[i]:
                        # calculate priority
                        priorities = self.get_priorities(self.pred_values_lst[i], self.search_values_lst[i])

                        # store current trajectory
                        self.game_histories[i].game_over()
                        self.put((self.game_histories[i], priorities))
                        self._free()
                        # reset the finished env and new a env
                        episodes_collected += 1
                        transitions_collected += len(self.game_histories[i])
                        self.log(i, temperature=temperature)
                        self.log(i, greedy_epsilon=greedy_epsilon)
                        self.reset_env(i)
                    elif len(self.game_histories[i]) > self.config.max_moves:
                        # discard this trajectory
                        self.reset_env(i)
                
                if episodes_collected >= num_envs :
                    break 

        return transitions_collected

    def close(self):
        self.replay_buffer = None
        self.shared_storage = None
        for env in self.envs:
            env.close()


@ray.remote
class RemoteDataWorker(DataWorker):

    def __init__(self, rank, config, replay_buffer: ActorHandle, shared_storage: ActorHandle):
        """Remote Data Worker for collecting data through self-play
        """
        assert isinstance(replay_buffer, ActorHandle), 'Must input RemoteReplayBuffer for RemoteDataWorker!'
        super().__init__(rank, config, replay_buffer, shared_storage)
        self.ray_store_obs = True  # put obs into ray memory

        formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s][%(filename)s>%(funcName)s] ==> %(message)s')
        file_path = os.path.join(config.exp_path, 'logs', 'root.log')
        self.logger = logging.getLogger('root')
        handler = logging.FileHandler(file_path, mode='a')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def _free(self):
        # save the game histories and clear the pool
        if len(self.trajectory_pool) >= self.pool_size:
            self.replay_buffer.save_pools.remote(self.trajectory_pool)
            del self.trajectory_pool[:]

    def _log_to_buffer(self, log_dict: dict):
        self.shared_storage.add_worker_logs.remote(log_dict)

    def _update_model_before_step(self):
        trained_steps = ray.get(self.shared_storage.get_counter.remote())
        if self.last_model_index // self.config.checkpoint_interval < trained_steps // self.config.checkpoint_interval:
            model_index, weights = ray.get(self.shared_storage.get_weights.remote())
            self.update_model(model_index, weights)

    def run_loop(self):

        start_training = False
        transitions_collected = 0

        # max transition to collect for this data worker
        max_transitions = self.config.total_transitions // self.config.data_actors

        while True:
            # ------------------ update training status ------------------
            trained_steps = ray.get(self.shared_storage.get_counter.remote())
            # (1) stop data-collecting when training finished
            if trained_steps >= self.config.training_steps + self.config.last_steps:
                time.sleep(10)
                break
            if not start_training:
                start_training = ray.get(self.shared_storage.get_start_signal.remote())
            # (2) balance training & selfplay
            if start_training and (transitions_collected / max_transitions) > (trained_steps / self.config.training_steps):
                # self-play is faster than training speed or finished
                target_trained_steps = math.ceil(transitions_collected / max_transitions * self.config.training_steps)
                self.logger.debug("(DataWorker{}) #{:<7} Wait for model updating...{}/{}".format(
                    self.rank, transitions_collected, trained_steps, target_trained_steps
                ))
                time.sleep(10)
                continue
            # -------------------------------------------------------------

            transitions_collected += self.run(start_training, trained_steps)

        self.close()
