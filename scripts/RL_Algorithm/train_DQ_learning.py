"""Script to train RL agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from omni.isaac.lab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RL_Algorithm.Algorithm.Double_Q_Learning import Double_Q_Learning
from tqdm import tqdm

from collections import deque
import yaml

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
import random

from omni.isaac.lab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml
from omni.isaac.lab_tasks.utils import get_checkpoint_path
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

# Import extensions to set up environment tasks
import CartPole.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with stable-baselines agent."""
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # directory for logging into
    log_dir = os.path.join("logs", "sb3", args_cli.task, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # ==================================================================== #
    # ========================= Can be modified ========================== #

    # hyperparameters
    num_of_action = 15
    action_range = [-5, 5]  # [min, max]
    discretize_state_weight = [10, 10, 5, 5]  # [pose_cart:int, pose_pole:int, vel_cart:int, vel_pole:int]
    learning_rate = 0.05
    n_episodes = 10000
    start_epsilon = 1.0
    epsilon_decay = 0.9999 # reduce the exploration over time
    final_epsilon = 0.1
    discount = 0.95

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

    # Store data to plot graph
    training_data = {
        "episode_rewards": [],
        "moving_avg_rewards": [],
        "steps_per_episode": [],
        "epsilon_values": []
    }
    moving_avg_window = deque(maxlen=100)  # For smoothing rewards

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    sum_reward = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
        
            for episode in tqdm(range(n_episodes)):
                obs, _ = env.reset()
                done = False
                cumulative_reward = 0
                step = 0

                while not done:
                    # agent stepping
                    action, action_idx = agent.get_action(obs)

                    # env stepping
                    next_obs, reward, terminated, truncated, _ = env.step(action)

                    reward_value = reward.item()
                    terminated_value = terminated.item() 
                    cumulative_reward += reward_value
                    
                    state = agent.discretize_state(obs)
                    next_state = agent.discretize_state(next_obs)
                    agent.update(state, action_idx, reward_value, next_state)

                    done = terminated or truncated
                    obs = next_obs
                    step += 1

                
                # Store episode reward and steps
                training_data["episode_rewards"].append(cumulative_reward)
                training_data["steps_per_episode"].append(step)
                training_data["epsilon_values"].append(agent.epsilon)

                # Compute moving average reward
                moving_avg_window.append(cumulative_reward)
                moving_avg_reward = sum(moving_avg_window) / len(moving_avg_window)
                training_data["moving_avg_rewards"].append(moving_avg_reward)
                
                sum_reward += cumulative_reward
                if episode % 100 == 0:
                    print("avg_score: ", sum_reward / 100.0)
                    sum_reward = 0
                    print(agent.epsilon)
                agent.decay_epsilon()
            
            # Save Q-Learning agent
            Algorithm_name = "Double_Q_learning"
            q_value_file = "name.json"
            full_path = os.path.join("q_value", Algorithm_name)
            agent.save_q_value(full_path, q_value_file)

            #save store data to yaml file
            version = 1
            while True:
                filename = f"training_results_{version}.yaml"
                file_path = os.path.join("Data", "Double_Q_learning", filename)
                
                if not os.path.exists(file_path):  # If the file does not exist, break
                    break
                version += 1  # Otherwise, increment and try again
            with open(file_path, "w") as yaml_file:
                yaml.dump(training_data, yaml_file, default_flow_style=False)
            
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        
        print("!!! Training is complete !!!")
        break
    # ==================================================================== #

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()