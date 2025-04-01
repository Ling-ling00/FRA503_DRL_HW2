"""Script to play RL agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from omni.isaac.lab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RL_Algorithm.Algorithm.Double_Q_Learning import Double_Q_Learning
from RL_Algorithm.Algorithm.Q_Learning import Q_Learning
from RL_Algorithm.Algorithm.SARSA import SARSA
from RL_Algorithm.Algorithm.MC import MC
from tqdm import tqdm

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from datetime import datetime
import json

from omni.isaac.lab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from omni.isaac.lab_tasks.utils import get_checkpoint_path
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

from omni.isaac.lab_tasks.utils.parse_cfg import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

# Import extensions to set up environment tasks
import CartPole.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

import numpy as np
import matplotlib.pyplot as plt

@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with stable-baselines agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,  #use_fabric=not args_cli.disable_fabric
    )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # ==================================================================== #
    # ========================= Can be modified ========================== #

    Algorithm_name = "MC"

    num_of_action = 11
    action_range = [-15, 15]  # [min, max]
    discretize_state_weight = [1, 21, 21, 1]  # [pose_cart:int, pose_pole:int, vel_cart:int, vel_pole:int]
    learning_rate = 0.1
    n_episodes = 5000
    start_epsilon = 0
    epsilon_decay = 0.9995 # reduce the exploration over time
    final_epsilon = 0
    discount = 0.7

    if Algorithm_name == "Double_Q_learning":
        agent = Double_Q_Learning(
            num_of_action=num_of_action,
            action_range=action_range,
            discretize_state_weight=discretize_state_weight,
            learning_rate=learning_rate,
            initial_epsilon=start_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount
        )
    elif Algorithm_name == "Q_learning":
        agent = Q_Learning(
            num_of_action=num_of_action,
            action_range=action_range,
            discretize_state_weight=discretize_state_weight,
            learning_rate=learning_rate,
            initial_epsilon=start_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount
        )
    elif Algorithm_name == "SARSA":
        agent = SARSA(
            num_of_action=num_of_action,
            action_range=action_range,
            discretize_state_weight=discretize_state_weight,
            learning_rate=learning_rate,
            initial_epsilon=start_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount
        )
    elif Algorithm_name == "MC":
        agent = MC(
            num_of_action=num_of_action,
            action_range=action_range,
            discretize_state_weight=discretize_state_weight,
            learning_rate=learning_rate,
            initial_epsilon=start_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount
        )

    task_name = str(args_cli.task).split('-')[0]  # Stabilize, SwingUp
    q_value_file = f"Part3/a_{num_of_action}_s_{discretize_state_weight[1]}/{Algorithm_name}_4900_{num_of_action}_15_1_{discretize_state_weight[1]}.json"
    full_path = os.path.join(f"q_value/{task_name}", Algorithm_name)
    agent.load_q_value(full_path, q_value_file)

    # reset environment
    obs, _ = env.reset()
    timestep = 0

    # === CONFIG ===
    log_observations = True
    max_timesteps = 300  # Stop logging after this many steps
    obs_log = []
    timestep = 0
    stop_collecting = False

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
        
            for episode in range(n_episodes):

                obs, _ = env.reset()
                done = False
                if stop_collecting:
                    env.close()
                    simulation_app.close()
                    break  # break inner loop

                while not done:
                    # agent stepping
                    action, action_idx = agent.get_action(obs)

                    # env stepping
                    next_obs, reward, terminated, truncated, _ = env.step(action)

                    done = terminated or truncated
                    obs = next_obs

                    if log_observations:
                        obs_log.append({
                            "timestep": timestep,
                            "cart_position": float(next_obs['policy'][0, 0].item()),
                            "cart_velocity": float(next_obs['policy'][0, 1].item()),
                            "pole_angle": float(next_obs['policy'][0, 2].item()),
                            "pole_angular_velocity": float(next_obs['policy'][0, 3].item()),
                            "action": float(action[0]) if isinstance(action, (list, tuple, np.ndarray)) else float(action)
                        })
        

                
                    timestep += 1
                    # Exit the play loop after recording one video
                    if timestep >= 1000:
                        logs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs"))
                        os.makedirs(logs_dir, exist_ok=True)
                        log_file = os.path.join(logs_dir, f"log_{Algorithm_name}_a_{num_of_action}_s_{discretize_state_weight[1]}.json")

                        with open(log_file, "w") as f:
                            json.dump(obs_log, f, indent=2)

                        print(f"\n✅ Saved {len(obs_log)} observations to: {log_file}")
                        stop_collecting = True
                        break

    # ==================================================================== #

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()