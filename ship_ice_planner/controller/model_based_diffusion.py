import numpy as np
import torch
from typing import Optional, List, Tuple
from tqdm import tqdm

from ship_ice_planner.controller.NRC_supply import NrcSupply
from ship_ice_planner.geometry.utils import Rxy


class ModelBasedDiffusionController(NrcSupply):
    """
    Model-Based Diffusion Controller following the reverse SDE approach from model-based-diffusion.
    Converted from JAX to PyTorch for ship ice navigation.
    """
    
    def __init__(self, 
                 horizon: int = 50,
                 num_samples: int = 2048,
                 num_diffusion_steps: int = 100,
                 temperature: float = 0.1,
                 beta0: float = 1e-4,
                 betaT: float = 1e-2,
                 seed: int = 0):
        """
        :param horizon: Prediction horizon (Hsample)
        :param num_samples: Number of samples per diffusion step (Nsample)
        :param num_diffusion_steps: Number of diffusion steps (Ndiffuse)
        :param temperature: Temperature for softmax weighting (temp_sample)
        :param beta0: Initial beta for diffusion schedule
        :param betaT: Final beta for diffusion schedule
        :param seed: Random seed
        """
        super().__init__()
        
        # Diffusion parameters
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_diffusion_steps = num_diffusion_steps
        self.temperature = temperature
        self.beta0 = beta0
        self.betaT = betaT
        
        # Control dimensions
        self.nu_dim = 3  # [surge, sway, yaw_rate]
        
        # Set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Compute diffusion schedule
        self._compute_diffusion_schedule()
        
        # Warm start for control sequence
        self.u_seq_mean = np.zeros((self.horizon, self.nu_dim))
        
        # Execution tracking
        self.execution_count = 0
        self.replan_interval = 5  # Replan every 5 steps
        
        # Override limits for full scale simulation (approximate)
        self.input_lims = [5.0, 1.0, 10.0]  # [m/s, m/s, deg/s]
        
    def _compute_diffusion_schedule(self):
        """Compute diffusion schedule parameters (betas, alphas, alpha_bar, sigmas)"""
        betas = np.linspace(self.beta0, self.betaT, self.num_diffusion_steps)
        alphas = 1.0 - betas
        alphas_bar = np.cumprod(alphas)  # Cumulative product
        
        # Sigma for noise schedule
        sigmas = np.sqrt(1 - alphas_bar)
        
        # Conditional sigmas (for reverse diffusion)
        alphas_bar_rolled = np.roll(alphas_bar, 1)
        alphas_bar_rolled[0] = 1.0
        Sigmas_cond = ((1 - alphas) * (1 - np.sqrt(alphas_bar_rolled)) / (1 - alphas_bar))
        sigmas_cond = np.sqrt(Sigmas_cond)
        sigmas_cond[0] = 0.0
        
        # Store as attributes
        self.betas = betas
        self.alphas = alphas
        self.alphas_bar = alphas_bar
        self.sigmas = sigmas
        self.sigmas_cond = sigmas_cond
        
        print(f"Initial sigma = {sigmas[-1]:.2e}")
    
    def rollout_trajectory(self, 
                          eta_start: np.ndarray, 
                          nu_start: np.ndarray,
                          u_seq: np.ndarray,
                          dt: float,
                          costmap=None,
                          goal_y: Optional[float] = None) -> Tuple[np.ndarray, float, List[np.ndarray]]:
        """
        Rollout a single trajectory given initial state and control sequence.
        Returns: (trajectory, total_reward, states_list)
        - trajectory: (horizon+1, 3) array of [x, y, psi]
        - total_reward: scalar reward (negative of total cost)
        - states_list: list of intermediate states for collision checking
        """
        eta = eta_start.copy()
        nu = nu_start.copy()
        trajectory = np.zeros((self.horizon + 1, 3))
        trajectory[0] = eta
        states_list = [eta.copy()]
        
        total_cost = 0.0
        
        for t in range(self.horizon):
            u_control = u_seq[t]
            
            # Dynamics step
            nu_deg = [nu[0], nu[1], np.rad2deg(nu[2])]
            nu_next_deg = self.dynamics(nu_deg[0], nu_deg[1], nu_deg[2], u_control)
            nu_next = [nu_next_deg[0], nu_next_deg[1], np.deg2rad(nu_next_deg[2])]
            
            # Integrate position
            u_g, v_g = Rxy(eta[2]) @ [nu_next[0], nu_next[1]]
            eta[0] += dt * u_g
            eta[1] += dt * v_g
            eta[2] = (eta[2] + dt * nu_next[2]) % (2 * np.pi)
            
            trajectory[t + 1] = eta
            states_list.append(eta.copy())
            nu = nu_next
            
            # Compute cost at this timestep
            # Costmap collision cost
            coll_cost = 0.0
            if costmap is not None:
                c_x = int(eta[0] * costmap.scale)
                c_y = int(eta[1] * costmap.scale)
                
                if 0 <= c_x < costmap.shape[1] and 0 <= c_y < costmap.shape[0]:
                    map_val = costmap.cost_map[c_y, c_x]
                    if map_val > 0:
                        coll_cost = map_val
                else:
                    coll_cost = 1000.0  # Out of bounds
            
            # Control effort (normalized)
            ctrl_cost = np.sum((u_control / 1000.0) ** 2) * 0.01
            
            total_cost += coll_cost + ctrl_cost
        
        # Goal-reaching reward (positive reward for making progress towards goal)
        goal_reward = 0.0
        if goal_y is not None:
            # Terminal reward: strong incentive for reaching/nearing goal
            final_y = trajectory[-1, 1]
            distance_to_goal = abs(final_y - goal_y)
            terminal_reward = 1000.0 / (1.0 + distance_to_goal)  # Large reward when close to goal
            
            # Progress reward: reward for making forward progress (y-increasing)
            y_progress = trajectory[-1, 1] - eta_start[1]
            progress_reward = max(0, y_progress) * 10.0  # Reward positive y movement
            
            goal_reward = terminal_reward + progress_reward
        
        # Reward is negative of cost + positive goal rewards
        total_reward = -total_cost + goal_reward
        
        return trajectory, total_reward, states_list
    
    def objective_function(self,
                          u_seq_batch: np.ndarray,
                          eta_start: np.ndarray,
                          nu_start: np.ndarray,
                          local_path: Optional[np.ndarray],
                          dt: float,
                          costmap=None,
                          goal_y: Optional[float] = None) -> np.ndarray:
        """
        Evaluate rewards for a batch of control sequences.
        Returns rewards (higher is better), not costs.
        
        :param u_seq_batch: (num_samples, horizon, 3) control sequences
        :param eta_start: (3,) initial pose [x, y, psi]
        :param nu_start: (3,) initial velocities [u, v, r]
        :param local_path: (horizon, 3) reference path or None
        :param dt: timestep
        :param costmap: CostMap object
        :return: (num_samples,) array of rewards
        """
        rewards = np.zeros(self.num_samples)
        
        for i in range(self.num_samples):
            trajectory, base_reward, states_list = self.rollout_trajectory(
                eta_start, nu_start, u_seq_batch[i], dt, costmap, goal_y=goal_y
            )
            
            # Add path tracking reward if local_path provided (but don't let it dominate goal-seeking)
            path_reward = 0.0
            if local_path is not None:
                for t in range(min(len(trajectory) - 1, len(local_path))):
                    pos_err = np.sum((trajectory[t+1, :2] - local_path[t, :2]) ** 2)
                    ang_err = trajectory[t+1, 2] - local_path[t, 2]
                    ang_err = (ang_err + np.pi) % (2 * np.pi) - np.pi
                    path_reward -= (pos_err + 50.0 * ang_err ** 2) * 0.1  # Reduced weight
            
            rewards[i] = base_reward + path_reward
        
        return rewards
    
    def reverse_diffusion_step(self,
                              i: int,
                              Ybar_i: np.ndarray,
                              eta_start: np.ndarray,
                              nu_start: np.ndarray,
                              local_path: Optional[np.ndarray],
                              dt: float,
                              costmap=None,
                              goal_y: Optional[float] = None) -> Tuple[np.ndarray, float]:
        """
        Single reverse diffusion step following the SDE formulation.
        
        :param i: Current diffusion step index
        :param Ybar_i: Current mean control sequence (horizon, nu_dim)
        :param eta_start: Initial pose
        :param nu_start: Initial velocities
        :param local_path: Reference path
        :param dt: Timestep
        :param costmap: CostMap object
        :return: (Ybar_im1, mean_reward) - updated mean and mean reward
        """
        # Convert Ybar_i to Yi (noisy version at step i)
        Yi = Ybar_i * np.sqrt(self.alphas_bar[i])
        
        # Sample from q_i: Y0s = eps * sigma_i + Ybar_i
        eps_u = np.random.normal(0, 1, (self.num_samples, self.horizon, self.nu_dim))
        Y0s = eps_u * self.sigmas[i] + Ybar_i
        Y0s = np.clip(Y0s, -2000, 2000)  # Clip to reasonable control limits
        
        # Evaluate trajectories and compute rewards
        rewards = self.objective_function(Y0s, eta_start, nu_start, local_path, dt, costmap, goal_y=goal_y)
        
        # Normalize rewards
        rew_mean = rewards.mean()
        rew_std = rewards.std()
        rew_std = rew_std if rew_std > 1e-4 else 1.0
        
        # Compute log probabilities (logp0)
        logp0 = (rewards - rew_mean) / rew_std / self.temperature
        
        # Softmax weights
        weights = np.exp(logp0 - np.max(logp0))  # Numerically stable
        weights = weights / weights.sum()
        
        # Update mean: Ybar = weighted average of Y0s
        Ybar = np.einsum('n,nij->ij', weights, Y0s)
        
        # Compute score function
        score = 1 / (1.0 - self.alphas_bar[i]) * (-Yi + np.sqrt(self.alphas_bar[i]) * Ybar)
        
        # Reverse step: Y_{i-1}
        Yim1 = 1 / np.sqrt(self.alphas[i]) * (Yi + (1.0 - self.alphas_bar[i]) * score)
        
        # Convert back to Ybar_{i-1}
        Ybar_im1 = Yim1 / np.sqrt(self.alphas_bar[i - 1])
        
        return Ybar_im1, rew_mean
    
    def DPcontrol(self,
                 pose: np.ndarray,
                 setpoint: np.ndarray,
                 dt: float,
                 nu: Optional[np.ndarray] = None,
                 local_path: Optional[np.ndarray] = None,
                 costmap=None,
                 goal_y: Optional[float] = None):
        """
        Main control function using model-based diffusion.
        
        :param pose: [x, y, psi] current pose
        :param setpoint: [x_ref, y_ref, psi_ref] target pose
        :param dt: timestep
        :param nu: [u, v, r] current velocities
        :param local_path: (N, 3) reference path
        :param costmap: CostMap object
        :return: Control input [u_control_0, u_control_1, u_control_2]
        """
        if nu is None:
            nu = np.array([0.0, 0.0, 0.0])
        
        # Check if we should replan
        if self.execution_count % self.replan_interval != 0 and len(self.u_seq_mean) > 0:
            idx = self.execution_count % self.replan_interval
            if idx < len(self.u_seq_mean):
                self.execution_count += 1
                return self.u_seq_mean[idx]
        
        # Replanning: run diffusion
        self.execution_count = 0
        
        # Prepare local path
        if local_path is None:
            local_path = np.tile(setpoint, (self.horizon, 1))
        else:
            local_path = np.array(local_path)
            # Ensure it matches horizon
            if len(local_path) != self.horizon:
                # Interpolate or repeat
                if len(local_path) < self.horizon:
                    # Repeat last point
                    last_point = local_path[-1:]
                    local_path = np.vstack([local_path, np.tile(last_point, (self.horizon - len(local_path), 1))])
                else:
                    # Take first horizon points
                    local_path = local_path[:self.horizon]
        
        # Warm start: shift previous plan
        shift = self.replan_interval
        if shift < self.horizon:
            self.u_seq_mean[:-shift] = self.u_seq_mean[shift:]
            self.u_seq_mean[-shift:] = self.u_seq_mean[-1]
        else:
            self.u_seq_mean[:] = self.u_seq_mean[-1] if len(self.u_seq_mean) > 0 else np.zeros((self.horizon, 3))
        
        # Initialize YN (noise at final step)
        YN = np.zeros((self.horizon, self.nu_dim))
        # Start from warm start instead of pure noise
        Ybar_i = self.u_seq_mean.copy()
        
        # Reverse diffusion process
        Ybars = []
        for i in tqdm(range(self.num_diffusion_steps - 1, 0, -1), desc="Diffusing", leave=False):
            Ybar_i, rew = self.reverse_diffusion_step(
                i, Ybar_i, pose, nu, local_path, dt, costmap, goal_y=goal_y
            )
            Ybars.append(Ybar_i)
        
        # Update mean control sequence
        self.u_seq_mean = Ybars[-1] if len(Ybars) > 0 else Ybar_i
        
        # Store predicted trajectory for visualization
        self.predicted_trajectory = self.simulate_trajectory(
            self.u_seq_mean, pose, nu, dt
        )
        
        # Store some sampled trajectories for visualization (from final step)
        self.sampled_trajectories_viz = []
        # Sample a few trajectories from the final distribution
        eps_final = np.random.normal(0, 1, (min(10, self.num_samples), self.horizon, self.nu_dim))
        Y0s_final = eps_final * self.sigmas[0] + self.u_seq_mean
        Y0s_final = np.clip(Y0s_final, -2000, 2000)
        for j in range(len(Y0s_final)):
            traj = self.simulate_trajectory(Y0s_final[j], pose, nu, dt)
            self.sampled_trajectories_viz.append(traj)
        
        self.execution_count += 1
        return self.u_seq_mean[0]
    
    def simulate_trajectory(self, u_seq: np.ndarray, eta_start: np.ndarray, nu_start: np.ndarray, dt: float) -> np.ndarray:
        """Helper to simulate a trajectory for visualization"""
        traj = np.zeros((self.horizon + 1, 3))
        traj[0] = eta_start
        eta = eta_start.copy()
        nu = nu_start.copy()
        
        for t in range(self.horizon):
            u_control = u_seq[t]
            nu_deg = [nu[0], nu[1], np.rad2deg(nu[2])]
            nu_next_deg = self.dynamics(nu_deg[0], nu_deg[1], nu_deg[2], u_control)
            nu_next = [nu_next_deg[0], nu_next_deg[1], np.deg2rad(nu_next_deg[2])]
            
            u_g, v_g = Rxy(eta[2]) @ [nu_next[0], nu_next[1]]
            eta[0] += dt * u_g
            eta[1] += dt * v_g
            eta[2] = (eta[2] + dt * nu_next[2]) % (2 * np.pi)
            
            nu = nu_next
            traj[t+1] = eta
            
        return traj

