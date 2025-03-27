import numpy as np
from collections import defaultdict
from enum import Enum
import os
import json
import torch


class ControlType(Enum):
    """
    Enum representing different control algorithms.
    """
    MONTE_CARLO = 1
    SARSA = 2
    Q_LEARNING = 3
    DOUBLE_Q_LEARNING = 4


class BaseAlgorithm():
    """
    Base class for reinforcement learning algorithms.

    Attributes:
        control_type (ControlType): The type of control algorithm used.
        num_of_action (int): Number of discrete actions available.
        action_range (list): Scale for continuous action mapping.
        discretize_state_scale (list): Scale factors for discretizing states.
        lr (float): Learning rate for updates.
        epsilon (float): Initial epsilon value for epsilon-greedy policy.
        epsilon_decay (float): Rate at which epsilon decays.
        final_epsilon (float): Minimum epsilon value allowed.
        discount_factor (float): Discount factor for future rewards.
        q_values (dict): Q-values for state-action pairs.
        n_values (dict): Count of state-action visits (for Monte Carlo method).
        training_error (list): Stores training errors for analysis.
    """

    def __init__(
        self,
        control_type: ControlType,
        num_of_action: int,
        action_range: list,  # [min, max]
        discretize_state_weight: list,  # [pose_cart:int, pose_pole:int, vel_cart:int, vel_pole:int]
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float,
    ):
        self.control_type = control_type
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.num_of_action = num_of_action
        self.action_range = action_range
        self.discretize_state_weight = discretize_state_weight

        self.q_values = defaultdict(lambda: np.zeros(self.num_of_action))
        self.n_values = defaultdict(lambda: np.zeros(self.num_of_action))
        self.training_error = []

        if self.control_type == ControlType.MONTE_CARLO:
            self.obs_hist = []
            self.action_hist = []
            self.reward_hist = []
        elif self.control_type == ControlType.DOUBLE_Q_LEARNING:
            self.qa_values = defaultdict(lambda: np.zeros(self.num_of_action))
            self.qb_values = defaultdict(lambda: np.zeros(self.num_of_action))
            self.na_values = defaultdict(lambda: np.zeros(self.num_of_action))
            self.nb_values = defaultdict(lambda: np.zeros(self.num_of_action))

    def discretize_state(self, obs: dict):
        """
        Discretize the observation state.

        Args:
            obs (dict): Observation dictionary containing policy states.

        Returns:
            Tuple[pose_cart:int, pose_pole:int, vel_cart:int, vel_pole:int]: Discretized state.
        """

        # ========= put your code here =========#
        policy_tensor = obs['policy']  # {'policy': tensor([[-0.2300,  0.0700, -0.1187, -0.1686]], device='cuda:0')}

        # Extracting the four values
        value = [0,0,0,0]
        value[0] = policy_tensor[0, 0].item()  # First value
        value[1] = policy_tensor[0, 1].item()  # Second value
        value[2] = policy_tensor[0, 2].item()   # Third value
        value[3] = policy_tensor[0, 3].item()   # Fourth value
        discrete_value = [0,0,0,0]

        bound = [[-3, 3],
                 [-np.deg2rad(24), np.deg2rad(24)],
                 [-5, 5],
                 [-5, 5]]

        for i in range(0,4):
            value[i] = max(bound[i][0], min(value[i], bound[i][1]))
            # Compute discrete bin
            discrete_value[i] = round((value[i] - bound[i][0]) * (self.discretize_state_weight[i] - 1) / (bound[i][1] - bound[i][0])) + 1

        return (discrete_value[0],
                discrete_value[1],
                discrete_value[2],
                discrete_value[3])
        # ======================================#

    def get_discretize_action(self, obs_dis) -> int:
        """
        Select an action using an epsilon-greedy policy.

        Args:
            obs_dis (tuple): Discretized observation.

        Returns:
            int: Chosen discrete action index.
        """
        # ========= put your code here =========#
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_of_action)
        elif self.control_type == ControlType.DOUBLE_Q_LEARNING:
            return int(np.argmax(self.qa_values[obs_dis]+self.qb_values[obs_dis]))
        return int(np.argmax(self.q_values[obs_dis]))
        # ======================================#
    
    def mapping_action(self, action):
        """
        Maps a discrete action in range [0, n] to a continuous value in [action_min, action_max].

        Args:
            action (int): Discrete action in range [0, n]
            n (int): Number of discrete actions
        
        Returns:
            torch.Tensor: Scaled action tensor.
        """
        # ========= put your code here =========#
        min_action, max_action = self.action_range
        action_step = (max_action - min_action) / (self.num_of_action - 1)
        action_value = min_action + action * action_step

        return torch.tensor([[action_value]], dtype=torch.float32)
        # ======================================#

    def get_action(self, obs) -> torch.tensor:
        """
        Get action based on epsilon-greedy policy.

        Args:
            obs (dict): The observation state.

        Returns:
            torch.Tensor, int: Scaled action tensor and chosen action index.
        """
        obs_dis = self.discretize_state(obs)
        action_idx = self.get_discretize_action(obs_dis)
        action_tensor = self.mapping_action(action_idx)
        return action_tensor, action_idx
    
    def decay_epsilon(self):
        """
        Decay epsilon value to reduce exploration over time.
        """
        # ========= put your code here =========#
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)
        # self.epsilon = max(self.final_epsilon, self.epsilon - (1-self.epsilon_decay))
        return self.epsilon
        # ======================================#

    def save_q_value(self, path, filename):
        """
        Save the model parameters to a JSON file.

        Args:
            path (str): Path to save the model.
            filename (str): Name of the file.
        """
        if self.control_type != ControlType.DOUBLE_Q_LEARNING:
            # Convert tuple keys to strings
            try:
                q_values_str_keys = {str(k): v.tolist() for k, v in self.q_values.items()}
            except:
                q_values_str_keys = {str(k): v for k, v in self.q_values.items()}
            # if self.control_type == ControlType.MONTE_CARLO:
            try:
                n_values_str_keys = {str(k): v.tolist() for k, v in self.n_values.items()}
            except:
                n_values_str_keys = {str(k): v for k, v in self.n_values.items()}
            
            # Save model parameters to a JSON file
            # if self.control_type == ControlType.MONTE_CARLO:
            model_params = {
                'q_values': q_values_str_keys,
                'n_values': n_values_str_keys
            }
            # else:
            #     model_params = {
            #         'q_values': q_values_str_keys,
            #     }
        else:
            # Convert tuple keys to strings
            try:
                qa_values_str_keys = {str(k): v.tolist() for k, v in self.qa_values.items()}
            except:
                qa_values_str_keys = {str(k): v for k, v in self.qa_values.items()}
            try:
                qb_values_str_keys = {str(k): v.tolist() for k, v in self.qb_values.items()}
            except:
                qb_values_str_keys = {str(k): v for k, v in self.qb_values.items()}    
            try:
                na_values_str_keys = {str(k): v.tolist() for k, v in self.na_values.items()}
            except:
                na_values_str_keys = {str(k): v for k, v in self.na_values.items()}
            try:
                nb_values_str_keys = {str(k): v.tolist() for k, v in self.nb_values.items()}
            except:
                nb_values_str_keys = {str(k): v for k, v in self.nb_values.items()}
            
            # Save model parameters to a JSON file
            model_params = {
                'qa_values': qa_values_str_keys,
                'qb_values': qb_values_str_keys,
                'na_values': na_values_str_keys,
                'nb_values': nb_values_str_keys
            }
        full_path = os.path.join(path, filename)
        with open(full_path, 'w') as f:
            json.dump(model_params, f)

            
    def load_q_value(self, path, filename):
        """
        Load model parameters from a JSON file.

        Args:
            path (str): Path where the model is stored.
            filename (str): Name of the file.

        Returns:
            dict: The loaded Q-values.
        """
        full_path = os.path.join(path, filename)        
        with open(full_path, 'r') as file:
            data = json.load(file)
            if self.control_type == ControlType.DOUBLE_Q_LEARNING:
                data_qa_values = data['qa_values']
                data_qb_values = data['qb_values']
                for state, action_values in data_qa_values.items():
                    state = state.replace('(', '')
                    state = state.replace(')', '')
                    tuple_state = tuple(map(float, state.split(', ')))
                    self.qa_values[tuple_state] = action_values.copy()
                for state, action_values in data_qb_values.items():
                    state = state.replace('(', '')
                    state = state.replace(')', '')
                    tuple_state = tuple(map(float, state.split(', ')))
                    self.qb_values[tuple_state] = action_values.copy()

            else:
                data_q_values = data['q_values']
                for state, action_values in data_q_values.items():
                    state = state.replace('(', '')
                    state = state.replace(')', '')
                    tuple_state = tuple(map(float, state.split(', ')))
                    self.q_values[tuple_state] = action_values.copy()
            if self.control_type == ControlType.MONTE_CARLO:
                data_n_values = data['n_values']
                for state, n_values in data_n_values.items():
                    state = state.replace('(', '')
                    state = state.replace(')', '')
                    tuple_state = tuple(map(float, state.split(', ')))
                    self.n_values[tuple_state] = n_values.copy()
            return self.q_values

