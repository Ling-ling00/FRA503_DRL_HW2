from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base import BaseAlgorithm, ControlType
from collections import defaultdict

class Double_Q_Learning(BaseAlgorithm):
    def __init__(
            self,
            num_of_action: int,
            action_range: list,
            discretize_state_weight: list,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float,
    ) -> None:
        """
        Initialize the Double Q-Learning algorithm.

        Args:
            num_of_action (int): Number of possible actions.
            action_range (list): Scaling factor for actions.
            discretize_state_weight (list): Scaling factor for discretizing states.
            learning_rate (float): Learning rate for Q-value updates.
            initial_epsilon (float): Initial value for epsilon in epsilon-greedy policy.
            epsilon_decay (float): Rate at which epsilon decays.
            final_epsilon (float): Minimum value for epsilon.
            discount_factor (float): Discount factor for future rewards.
        """
        super().__init__(
            control_type=ControlType.DOUBLE_Q_LEARNING,
            num_of_action=num_of_action,
            action_range=action_range,
            discretize_state_weight=discretize_state_weight,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )
        self.q_values2 = defaultdict(lambda: np.zeros(self.num_of_action))
        
    def update(self, state, action, reward, next_state):
        """
        Update Q-values using Double Q-Learning.

        This method applies the Double Q-Learning update rule to improve policy decisions by updating the Q-table.
        """
        if np.random.random() < 0.5:
            # Update Q_1
            max_next_q_value = np.max(self.q_values2[next_state]) if next_state is not None else 0
            q_value = self.q_values[state][action]
            self.q_values[state][action] = q_value + self.lr * (reward + (self.discount_factor * max_next_q_value) - q_value)
        else:
            # Update Q_2
            max_next_q_value = np.max(self.q_values[next_state]) if next_state is not None else 0
            q_value = self.q_values2[state][action]
            self.q_values2[state][action] = q_value + self.lr * (reward + (self.discount_factor * max_next_q_value) - q_value)
