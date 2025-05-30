import argparse
import os
import torch
import numpy as np
import logging
from types import SimpleNamespace
import importlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    GameConfig = importlib.import_module("config.smac").GameConfig
    BaseNet = importlib.import_module("core.model").BaseNet
    CustomStarCraft2Env = importlib.import_module("config.smac.mappo_smac.StarCraft2_Env").StarCraft2Env
    SMACWrapper = importlib.import_module("config.smac.env_wrapper").SMACWrapper
    logger.info("Successfully imported project modules.")
except ImportError as e:
    logger.error(f"Failed to import project modules: {e}. Ensure script runs from root and paths are correct.")
    exit(1)

# Most of this does not apply but is necessary to run
def create_eval_config_args(target_map, source_map, replay_dir):
    """Creates a comprehensive args namespace for GameConfig during evaluation."""
    replay_base = replay_dir or os.path.join(os.getcwd(), "results_replay_transfer")
    exp_path = os.path.join(replay_base, "smac", f"{source_map}_to_{target_map}", "replay_run")
    os.makedirs(exp_path, exist_ok=True)
    logger.info(f"Replays will be saved in: {exp_path}")

    args_dict = {
        "opr": "test",
        "case": "smac",
        "env_name": target_map,
        "exp_name": f"replay_gen_{source_map}_on_{target_map}",
        "seed": 1001,
        "discount": 0.997,
        "result_dir": replay_base,
        "use_wandb": False,
        "num_gpus": 1,
        "num_cpus": 1,
        "object_store_memory": 0,
        "selfplay_on_gpu": True,
        "data_actors": 1,
        "num_pmcts": 1,
        "checkpoint_interval": 1000,
        "total_transitions": 100000,
        "start_transitions": 300,
        "use_priority": False,
        "use_max_priority": False,
        "use_change_temperature": False,
        "eps_start": 0.0,
        "eps_end": 0.0,
        "eps_annealing_time": 1000,
        "use_priority_refresh": False,
        "refresh_actors": 1,
        "refresh_interval": 100,
        "refresh_mini_size": 256,
        "num_simulations": 50,
        "pb_c_base": 19652,
        "pb_c_init": 1.25,
        "tree_value_stat_delta_lb": 0.01,
        "root_dirichlet_alpha": 0.3,
        "root_exploration_fraction": 0.25,
        "sampled_action_times": 10, 
        "mcts_rho": 0.75,
        "mcts_lambda": 0.8,
        "train_on_gpu": True,
        "training_steps": 10000,
        "last_steps": 10000,
        "batch_size": 256,
        "num_unroll_steps": 5,
        "max_grad_norm": 5.0,
        "reward_loss_coeff": 1.0,
        "value_loss_coeff": 0.25,
        "policy_loss_coeff": 1.0,
        "use_consistency_loss": False,
        "consistency_coeff": 2.0,
        "awac_lambda": 3.0, 
        "adv_clip": 3.0,
        "PG_type": 'sharp',
        "lr": 1e-4,
        "lr_adjust_func": "const",
        "opti_eps": 1e-5,
        "weight_decay": 0,
        "reanalyze_on_gpu": True,
        "reanalyze_actors": 1,
        "reanalyze_update_actors": 0,
        "td_steps": 5,
        "target_value_type": "pred-re",
        "revisit_policy_search_rate": 1.0,
        "use_off_correction": True,
        "auto_td_steps_ratio": 0.3,
        "target_model_interval": 200,
        "test_interval": 1000,
        "test_episodes": 1,
        "pretrained_model_path": None,
        "use_mcts_test": False,
        "save_interval": 10000,
        "log_interval": 100,
        "use_augmentation": False,
        "augmentation": ["shift", "intensity"],
        "value_transform_type": "vector",
        "ppo_loss_proportion": 0.5,
        "stacked_observations": 1,

    }
    args_dict['our_exp_path'] = exp_path
    return SimpleNamespace(**args_dict)

def adapt_weights(source_sd, target_model, device, target_config):
    """Adapts source state_dict to target model, handling shape mismatches."""
    target_sd = target_model.state_dict()
    adapted_sd = {}
    logger.info("Starting state_dict adaptation...")

    for key, t_param in target_sd.items():
        if key not in source_sd:
            logger.warning(f"'{key}' in target but not source. Using target init.")
            adapted_sd[key] = t_param.clone().to(device)
            continue

        s_param = source_sd[key].to(device)
        s_shape, t_shape = s_param.shape, t_param.shape

        if s_shape == t_shape:
            adapted_sd[key] = s_param
            continue

        logger.info(f"Adapting '{key}': {s_shape} -> {t_shape}")
        new_p = t_param.clone().to(device)

        try:
            # --- Rules for adapting layers ---
            # Rule 1: Input Feature Norm (1D)
            if "representation_network.feature_norm" in key:
                c = min(s_shape[0], t_shape[0])
                new_p[:c] = s_param[:c]
                if t_shape[0] > s_shape[0]:
                    new_p[c:].fill_(1.0 if "weight" in key else 0.0)
            # Rule 2: Input MLP (2D - Input Dim Change)
            elif "representation_network.mlp.0.weight" in key:
                c_in = min(s_shape[1], t_shape[1])
                new_p[:, :c_in] = s_param[:, :c_in]
                if t_shape[1] > s_shape[1]: new_p[:, c_in:].fill_(0.0)
            # Rule 3: Attention/Dynamics with Action Input (2D - Input Dim Change)
            elif "dynamics_network.attention_stack.0.weight" in key or \
                 "dynamics_network.fc_dynamic.0.weight" in key:
                h_dim = 128 # *** ASSUMPTION: Hidden/Att dim ***
                prefix_dim = h_dim if "attention" in key else h_dim * 2
                s_a = s_shape[1] - prefix_dim
                t_a = t_shape[1] - prefix_dim
                c_a = min(s_a, t_a)
                new_p[:, :prefix_dim] = s_param[:, :prefix_dim]
                new_p[:, prefix_dim:prefix_dim + c_a] = s_param[:, prefix_dim:prefix_dim + c_a]
                if t_a > s_a: new_p[:, prefix_dim + c_a:].fill_(0.0)
            # Rule 4: Policy Head (Output Dim Change)
            elif "prediction_network.fc_policy" in key:
                c_out = min(s_shape[0], t_shape[0])
                if len(s_shape) == 2: # Weight
                    new_p[:c_out, :] = s_param[:c_out, :]
                    if t_shape[0] > s_shape[0]: new_p[c_out:, :].fill_(0.0)
                else: # Bias
                    new_p[:c_out] = s_param[:c_out]
                    if t_shape[0] > s_shape[0]: new_p[c_out:].fill_(0.0)
            # Rule 5: Agent Projection (Input Dim Change - Replicates weights)
            elif "projection_network.projection.0.weight" in key:
                repr_dim = 128 # *** ASSUMPTION: Representation dim ***
                t_agents = target_config.num_agents
                if s_shape[1] < repr_dim or t_shape[1] != t_agents * repr_dim:
                    raise ValueError("Cannot apply projection rule.")
                logger.warning(f"Replicating first agent weights for '{key}'. This is a heuristic.")
                first_agent_w = s_param[:, :repr_dim]
                for i in range(t_agents):
                    new_p[:, i * repr_dim : (i + 1) * repr_dim] = first_agent_w
            else:
                raise ValueError("Unhandled mismatch.")

            adapted_sd[key] = new_p

        except Exception as e:
            logger.error(f"Failed to adapt '{key}': {e}. Using target init.")
            adapted_sd[key] = t_param.clone().to(device) # Fallback

    logger.info("State_dict adaptation finished.")
    return adapted_sd

def run_episode(config, model, env, device):
    """Runs a single evaluation episode."""
    logger.info(f"Starting episode on map '{config.env_name}'...")
    try:
        # Initialization
        obs_wrapped = env.reset()
        terminated = False
        episode_reward = 0
        step = 0
        n_agents = config.num_agents

        obs_squeezed = obs_wrapped.squeeze(axis=(2, 3))
        obs_tensor = torch.tensor(obs_squeezed, dtype=torch.float32).unsqueeze(0).to(device)
        
        hidden_state = None
        last_actions_tensor = torch.zeros((1, n_agents), dtype=torch.long).to(device)

        with torch.no_grad():
            network_output = model.initial_inference(obs_tensor)
            policy_logits_np = network_output.policy_logits 
            policy_logits = torch.from_numpy(policy_logits_np).to(device) 
            hidden_state = network_output.hidden_state

            # Main Episode Loop
            while not terminated:
                avail_actions_list = env.legal_actions()

                if policy_logits.shape[1] != n_agents:
                    raise ValueError("Policy logits agent dim != n_agents.")

                actions = []
                for agent_id in range(n_agents):
                    agent_avail = torch.tensor(avail_actions_list[agent_id], dtype=torch.bool).to(device)
                    agent_logits = policy_logits[0, agent_id]

                    if agent_logits.shape[0] != agent_avail.shape[0]:
                         raise ValueError(f"Logits ({agent_logits.shape}) vs Avail ({agent_avail.shape}) mismatch for agent {agent_id}")

                    # Mask Illegal Actions
                    mask_to_fill = torch.logical_not(agent_avail)
                    agent_logits.masked_fill_(mask_to_fill, -float('inf')) 

                    # Select Best Action
                    actions.append(torch.argmax(agent_logits).item())

                last_actions_tensor = torch.tensor(actions, dtype=torch.long).unsqueeze(0).to(device)
                obs_wrapped, reward, terminated, info = env.step(actions)
                episode_reward += reward
                step += 1

                if terminated:
                    win = info.get("battle_won", False)
                    logger.info(f"Episode ended @ step {step}. Win: {win}. Reward: {episode_reward:.2f}")
                    break
                
                # Prepare for Next Step (Recurrent Inference)
                network_output = model.recurrent_inference(hidden_state, last_actions_tensor)
                policy_logits_np = network_output.policy_logits
                policy_logits = torch.from_numpy(policy_logits_np).to(device) 
                hidden_state = network_output.hidden_state

    except Exception as e:
        logger.error(f"Error during episode: {e}", exc_info=True)
    finally:
        if 'env' in locals() and hasattr(env, 'close'):
            env.close()
            logger.info("Environment closed.")

def main(cli_args):
    """Orchestrates loading and running the evaluation."""
    device = torch.device("cuda" if torch.cuda.is_available() and not cli_args.cpu else "cpu")
    logger.info(f"Using device: {device}")
    
    config_args = None 

    try:
        config_args = create_eval_config_args( 
            cli_args.target_map,
            cli_args.source_map,
            cli_args.replay_dir
        )
        if cli_args.cpu:
            config_args.train_on_gpu = False
            config_args.selfplay_on_gpu = False
            config_args.reanalyze_on_gpu = False

        # 2. Load Config & Model Structure (Target)
        logger.info(f"Loading config for target map: {config_args.env_name}")
        config = GameConfig(config_args)
        
        our_desired_path = config_args.our_exp_path
        config.exp_path = our_desired_path
        config.model_dir = os.path.join(our_desired_path, 'model') # Also update model_dir
        config.model_path = os.path.join(our_desired_path, 'model.p') # Also update model_path
        logger.info(f"Manually set config.exp_path to: {config.exp_path}")

        model = config.get_uniform_network()
        logger.info(f"Target Config: Obs={config.obs_shape}, Actions={config.action_space_size}, Agents={config.num_agents}")

        # 3. Load Source Weights & Adapt
        logger.info(f"Loading source weights from: {cli_args.model_path}")
        if not os.path.exists(cli_args.model_path):
             raise FileNotFoundError(f"Model file not found: {cli_args.model_path}")
        source_state_dict = torch.load(cli_args.model_path, map_location="cpu")

        adapted_state_dict = adapt_weights(source_state_dict, model, device, config)
        model.load_state_dict(adapted_state_dict, strict=True) 
        model.to(device).eval()
        logger.info("Adapted weights loaded successfully.")

        # 4. Initialize Environment (Target) & Run
        logger.info(f"Initializing environment: {config.env_name}")
        # Now, new_game should use the manually set config.exp_path
        env = config.new_game(seed=config.seed, save_video=True) 
        run_episode(config, model, env, device)

    except FileNotFoundError as e:
        logger.error(f"{e}")
    except Exception as e:
        logger.error(f"An error occurred in main: {e}", exc_info=True)

    if config_args:
        logger.info(f"Evaluation finished. Check {config_args.our_exp_path} for replays.")
    else:
        logger.info("Evaluation finished with early error.")

# --- Argument Parsing ---
if __name__ == "__main__":
    print("DEBUG: Script started.") # <<< ADD
    parser = argparse.ArgumentParser(description="Evaluate MAZero model on a different SMAC map.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the source model (.p file).")
    parser.add_argument("--source_map", type=str, default="3m", help="Map the model was trained on.")
    parser.add_argument("--target_map", type=str, default="4m", help="Map to evaluate on.")
    parser.add_argument("--replay_dir", type=str, default=None, help="Base directory for replays.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU evaluation.")

    print("DEBUG: Parsing arguments...") # <<< ADD
    args = parser.parse_args()
    print(f"DEBUG: Arguments parsed: {args}") # <<< ADD

    print("DEBUG: Calling main...") # <<< ADD
    main(args)
    print("DEBUG: Main finished.") # <<< ADD