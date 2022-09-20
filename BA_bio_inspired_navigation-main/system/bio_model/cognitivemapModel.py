import numpy as np
import os

# Concept based on Erdem 2012. For detailed explanations please refer to thesis or paper.

# Decisions
# Only one place cell is active at the same time -> winner takes it all
# Binary connections in cognitive map between two topology cells


class CognitiveMapNetwork:
    """The CognitiveMapNetwork keeps track of all recency, topology and reward cells"""
    def __init__(self, dt, from_data=False):
        if not from_data:
            self.recency_cells = np.array([])  # array of firing values between 0 and 1; 1 where the agent is
            self.topology_cells = np.zeros((1, 1))  # matrix of connections, size (#pc x #pc)
            self.reward_cells = np.array([])  # array of firing values between 0 and 1; 1 where the goal is
        else:
            self.topology_cells = np.load("data/cognitive_map/topology_cells.npy")
            self.reward_cells = np.load("data/cognitive_map/reward_cells.npy")
            self.recency_cells = np.zeros_like(self.reward_cells)
            self.recency_cells[0] = 1

        self.dt = dt

        epsilon = 2  # parameters tuned for velocity and environment size
        lam = 0.01 / dt  # parameters tuned for velocity and environment size
        self.decay_rate = epsilon**(-lam * dt)  # determines how quickly recency cell firing decays
        self.recency_threshold = 0.5  # determines when we still consider an recency cell as active

        self.prior_idx_pc_firing = 0  # which pc was firing in last time step

        self.active_threshold = 0.85  # determines when we consider a place cell to be active

    def add_cortical_column(self, reward):
        """Adds a set of three prefrontal cortex cells to network. Called when a new place cell was created."""

        # Add a recency cell to the end, currently active
        self.recency_cells = np.append(self.recency_cells, 1)

        # Extend topology cell array by a row and column, no connections have formed yet
        n = len(self.recency_cells)
        reference_array = np.zeros((n, n))
        reference_array[:self.topology_cells.shape[0], :self.topology_cells.shape[1]] = self.topology_cells
        self.topology_cells = reference_array

        # Add a reward cell to the end, reward value depends on if an reward has been found
        self.reward_cells = np.append(self.reward_cells, reward)

    def compute_reward_spiking(self, pc_firing):
        """Determine which place cells are active and multiply with reward value"""
        pc_firing = np.where(np.array(pc_firing) > self.active_threshold, pc_firing, 0)  # Check for active pc
        rewards = self.reward_cells * pc_firing  # Multiply with reward spiking
        idx_pc_active = np.argmax(rewards)
        reward = np.max(rewards)
        return [reward, idx_pc_active]  # Return highest reward and idx of pc

    def track_movement(self, pc_firing, created_new_pc, reward):
        """Keeps track of current place cell firing and creation of new place cells"""

        if created_new_pc:
            self.add_cortical_column(reward)

        idx_pc_active = np.argmax(pc_firing)  # max one place cell is considered as active
        pc_active = np.max(pc_firing)

        # update recency cells
        self.recency_cells = self.recency_cells * self.decay_rate
        if pc_active > self.active_threshold:
            # set recency cell of current place cell to 1
            self.recency_cells[idx_pc_active] = 1

        # Check if we have entered a new place cell
        if created_new_pc:
            entered_different_pc = True
        elif pc_active > self.active_threshold and self.prior_idx_pc_firing != idx_pc_active:
            entered_different_pc = True
        else:
            entered_different_pc = False

        if entered_different_pc:
            # update topology cells, refer to thesis for formulas
            prior_visited = np.heaviside(self.recency_cells - self.recency_threshold, 1)
            currently_visited = np.heaviside(self.recency_cells - 1, 1)
            new_connections = np.outer(prior_visited, currently_visited)
            # save bilateral connections
            self.topology_cells = np.maximum(self.topology_cells, new_connections)
            self.topology_cells = np.maximum(self.topology_cells, np.transpose(new_connections))

            # update reward cells, refer to thesis for formulas
            reward_cells = np.where(self.reward_cells == 1, 1, 0)
            reward_cells_prior = reward_cells
            for t in range(1, 16):
                reward_decay = 1 / (t + 1)
                reward_cells = np.heaviside(np.dot(reward_cells_prior, self.topology_cells), 0)
                reward_cells = np.maximum(reward_cells * reward_decay, reward_cells_prior)
                reward_cells_prior = reward_cells
            self.reward_cells = reward_cells

            self.prior_idx_pc_firing = idx_pc_active

    def save_cognitive_map(self, filename=""):
        directory = "data/cognitive_map/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save("data/cognitive_map/recency_cells" + filename + ".npy", self.recency_cells)
        np.save("data/cognitive_map/topology_cells" + filename + ".npy", self.topology_cells)
        np.save("data/cognitive_map/reward_cells" + filename + ".npy", self.reward_cells)
