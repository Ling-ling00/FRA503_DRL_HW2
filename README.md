# CartPole
 
## Overview
This repository contains files related to the `CartPole` task, which will be used for Homework 2 and Homework 3. It includes environment configurations, RL algorithms, and training scripts to support reinforcement learning experiments.


The repository provided in this homework is a custom IsaacLab extension, created from [IsaacLabExtensionTemplate](https://github.com/isaac-sim/IsaacLabExtensionTemplate). Please refer to this repository for more detailed information.

## Installation

- navigate to the `CartPole_4.2.0/` directory by running:

    ```
    cd CartPole_4.2.0/
    ```
- Using a python interpreter (conda) that has Isaac Lab installed, install the library

    ```
    python -m pip install -e ./exts/CartPole
    ```

- Verify that the extension is correctly installed by running the following command to print all the available environments in the extension:

    ```
    python scripts/list_envs.py
    ```
## Repository organization
This repository is an IsaacLab extension for training reinforcement learning (RL) agents on the CartPole task. It includes environment configurations, RL algorithms, and training scripts.

```
CARTPOLE
├── exts/CartPole
│   ├── CartPole
│       ├── tasks
│           ├── cartpole
│               ├── agents
│               ├── mdp
│               │   ├── __init__.py
│               │   ├── actions.py
│               │   ├── events.py
│               │   ├── observation.py
│               │   ├── rewards.py
│               │   └── terminations.py
│               ├── __init__.py
│               ├── stabilize_cartpole_env_cfg.py
│               └── swing_up_cartpole_env_cfg.py
│               
│  
├── q_value
│   ├── Stabilize
│   │   ├── tasks
│   │   ├── MC
│   │   ├── Q_Learning
│   │   └── SARSA
│   ├── SwingUp
│  
├── RL_Algorithm
│   ├── Algorithm
│   │   ├── Double_Q_Learning
│   │   ├── MC
│   │   ├── Q_Learning
│   │   └── SARSA
│   └── RL_base.py
│
└── scripts
    └── RL_Algorithm
        ├── play.py
        ├── random_action.py
        └── train.py
```

### Descriptions

- **exts/CartPole:** Contains the core elements of the CartPole environments.

    - **mdp:** Implements key components of the Markov Decision Process (MDP) which includes actions, events, observations, rewards, and termination conditions.

    - **__init__.py:** Contains the gym registry
    code that registers your environments with the `OpenAI Gym interface`. This registration makes your environments compatible with standard RL libraries and algorithms (please consults this [tutorial](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/register_rl_env_gym.html#using-the-gym-registry) for more information).

    - **stabilize_cartpole_env_cfg.py / swing_up_cartpole_env_cfg.py:** `Manager-Based RL Environments` configuration for the stabilization task and swing-up task:

        - `Scene`

        - `Action` 

        - `Observation`

        - `Event`

        - `Reward`

        - `Termination`

- **q_value:** Stores the trained `Q-value tables` as `JSON files`. Each subdirectory corresponds to a different `Q-value tables` learned by algorithms.

- **RL_Algorithm:** This is where you'll implement your reinforcement learning algorithms for Homework 2:

    - **Algorithm:** Contains separate implementations for each RL approach:

        - `MC`

        - `SARSA` 

        - `Q-Learning`

        - `Double Q-Learning`

    - **RL_base.py:** Provides the foundation classes with common methods such as:

        - `get_action`

        - `decay_epsilon` 

        - `save_model`

        - `load_model`


- **scripts/RL_Algorithm:** Contains executable scripts for:

    - **train.py:** Runs the training process for your selected algorithm against a specific environment.

    - **play.py:** Demonstrates the performance of a trained agent using saved Q-values.
 
    - **random_action.py:** Executes random actions in the environment to verify package installation.

## Verifying CartPole installation

### 1. Stabilizing Cart-Pole Task

```
python scripts/RL_Algorithm/random_action.py --task Stabilize-Isaac-Cartpole-v0
```

### 2. Swing-up Cart-Pole Task

```
python scripts/RL_Algorithm/random_action.py --task SwingUp-Isaac-Cartpole-v0
```

