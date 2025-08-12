# PHASE 1 - STEP 4: REINFORCE AGENT
# This implements the REINFORCE algorithm - the learning mechanism for our poker agent

"""
REINFORCE ALGORITHM EXPLANATION:

REINFORCE is a Monte Carlo Policy Gradient method. Here's how it works:

1. PLAY EPISODES: Complete full poker hands from start to finish
2. CALCULATE RETURNS: For each action, calculate the total reward that followed
3. UPDATE POLICY: Increase probability of actions that led to good outcomes

Mathematical Intuition:
- Good outcome (won chips) → Increase probability of actions that led to this
- Bad outcome (lost chips) → Decrease probability of actions that led to this

The key insight: We learn from complete episodes, not individual actions
This is perfect for poker because:
- Individual actions might seem bad but lead to good overall outcomes
- We need to consider the full sequence of play
- Bluffs might lose most of the time but be profitable overall

REINFORCE Formula:
∇J(θ) = E[∇log π(a|s,θ) * G_t]
Where:
- θ = neural network parameters
- π(a|s,θ) = policy (action probability given state)
- G_t = return (total discounted reward from time t)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt

class REINFORCEAgent:
    """
    REINFORCE Agent for Poker
    
    This agent learns to play poker by:
    1. Playing complete hands
    2. Recording actions and rewards
    3. Updating policy to increase probability of successful actions
    """
    
    def __init__(self, 
                 policy_network: nn.Module,
                 learning_rate: float = 0.01,
                 gamma: float = 0.99):
        """
        Initialize the REINFORCE agent
        
        Args:
            policy_network: The neural network that defines our strategy
            learning_rate: How fast we update the policy
            gamma: Discount factor (how much we value future rewards)
        """
        self.policy_network = policy_network
        self.gamma = gamma
        
        # Optimizer for updating network weights
        self.optimizer = optim.Adam(policy_network.parameters(), lr=learning_rate)
        
        # Storage for episode data
        self.episode_log_probs = []  # Log probabilities of actions taken
        self.episode_rewards = []    # Rewards received
        self.episode_states = []     # States encountered
        self.episode_actions = []    # Actions taken
        
        # Training statistics
        self.episode_returns = []    # Total return for each episode
        self.episode_lengths = []    # Length of each episode
        self.policy_losses = []      # Policy loss over time
        
        print(f"🤖 REINFORCE Agent created:")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Discount factor (gamma): {gamma}")
        print(f"   Optimizer: Adam")
    
    def select_action(self, state: np.ndarray, legal_actions: List[int]) -> int:
        """
        Select an action using the current policy
        
        This is where the agent "thinks" about what to do in a poker situation
        
        Args:
            state: Current game state (cards, betting, etc.)
            legal_actions: Valid actions in this situation
            
        Returns:
            action: The action the agent chooses to take
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get action from policy network
        action, log_prob = self.policy_network.select_action(
            state_tensor.squeeze(), legal_actions
        )
        
        # Store information for later learning
        self.episode_log_probs.append(log_prob)
        self.episode_states.append(state)
        self.episode_actions.append(action)
        
        return action
    
    def store_reward(self, reward: float):
        """
        Store the reward received from the environment
        
        In poker, this is usually:
        - Positive for winning chips
        - Negative for losing chips
        - Zero for ongoing play
        """
        self.episode_rewards.append(reward)
    
    def _calculate_returns(self) -> List[float]:
        """
        Calculate discounted returns for each action in the episode
        
        RETURN CALCULATION EXPLANATION:
        
        Return G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...
        
        Where:
        - r_t = immediate reward at time t
        - γ = discount factor (0.99)
        - γ < 1 means future rewards are worth slightly less
        
        Example in poker:
        - Action 1: Call (r=0), Action 2: Raise (r=0), Final: Win (+2)
        - Return for Action 1: 0 + 0.99*0 + 0.99²*2 = 1.96
        - Return for Action 2: 0 + 0.99*2 = 1.98
        
        This means both actions get credit for the final win!
        """
        returns = []
        G = 0  # Initialize return
        
        # Work backwards through the episode
        for reward in reversed(self.episode_rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)  # Insert at beginning
        
        return returns
    
    def _normalize_returns(self, returns: List[float]) -> torch.Tensor:
        """
        Normalize returns to improve training stability
        
        WHY NORMALIZE?
        - Reduces variance in gradient updates
        - Helps training converge faster
        - Standard practice in policy gradient methods
        """
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        
        # Normalize: (x - mean) / std
        if len(returns) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-9)
        
        return returns_tensor
    
    def update_policy(self) -> float:
        """
        Update the policy based on the episode experience
        
        This is the heart of REINFORCE learning!
        
        POLICY GRADIENT THEOREM:
        ∇J(θ) = E[∇log π(a|s,θ) * G_t]
        
        In plain English:
        - If action A led to good return G, increase probability of A
        - If action B led to bad return G, decrease probability of B
        - The magnitude of change depends on how good/bad the return was
        """
        
        if len(self.episode_log_probs) == 0:
            return 0.0  # No episode to learn from
        
        # Step 1: Calculate returns for each action
        returns = self._calculate_returns()
        returns_tensor = self._normalize_returns(returns)
        
        # Step 2: Calculate policy loss
        policy_loss = []
        
        for log_prob, G in zip(self.episode_log_probs, returns_tensor):
            # Policy gradient: -log_prob * return
            # Negative sign because we want to maximize return (minimize negative return)
            policy_loss.append(-log_prob * G)
        
        # Sum all losses
        policy_loss = torch.cat(policy_loss).sum()
        
        # Step 3: Update network parameters
        self.optimizer.zero_grad()  # Clear previous gradients
        policy_loss.backward()      # Calculate gradients
        
        # Gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()       # Update weights
        
        # Step 4: Store statistics and clear episode data
        total_return = sum(self.episode_rewards)
        self.episode_returns.append(total_return)
        self.episode_lengths.append(len(self.episode_rewards))
        self.policy_losses.append(policy_loss.item())
        
        # Clear episode data for next episode
        self.episode_log_probs = []
        self.episode_rewards = []
        self.episode_states = []
        self.episode_actions = []
        
        return policy_loss.item()
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get training statistics for monitoring progress
        """
        if not self.episode_returns:
            return {}
        
        recent_episodes = min(100, len(self.episode_returns))
        recent_returns = self.episode_returns[-recent_episodes:]
        recent_lengths = self.episode_lengths[-recent_episodes:]
        
        return {
            'total_episodes': len(self.episode_returns),
            'avg_return': np.mean(recent_returns),
            'avg_length': np.mean(recent_lengths),
            'win_rate': np.mean([r > 0 for r in recent_returns]),
            'best_return': max(self.episode_returns),
            'worst_return': min(self.episode_returns),
            'total_return': sum(self.episode_returns)
        }
    
    def plot_training_progress(self):
        """
        Visualize training progress
        """
        if len(self.episode_returns) < 10:
            print("Not enough episodes to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot 1: Episode returns
        axes[0, 0].plot(self.episode_returns)
        axes[0, 0].set_title('Episode Returns')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Return')
        axes[0, 0].grid(True)
        
        # Plot 2: Running average returns
        window = min(50, len(self.episode_returns) // 4)
        if window > 1:
            running_avg = []
            for i in range(len(self.episode_returns)):
                start_idx = max(0, i - window + 1)
                avg = np.mean(self.episode_returns[start_idx:i+1])
                running_avg.append(avg)
            
            axes[0, 1].plot(running_avg)
            axes[0, 1].set_title(f'Running Average Returns (window={window})')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Average Return')
            axes[0, 1].grid(True)
        
        # Plot 3: Episode lengths
        axes[1, 0].plot(self.episode_lengths)
        axes[1, 0].set_title('Episode Lengths')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Number of Actions')
        axes[1, 0].grid(True)
        
        # Plot 4: Policy losses
        if self.policy_losses:
            axes[1, 1].plot(self.policy_losses)
            axes[1, 1].set_title('Policy Loss')
            axes[1, 1].set_xlabel('Update Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'policy_network_state_dict': self.policy_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_returns': self.episode_returns,
            'episode_lengths': self.episode_lengths,
            'policy_losses': self.policy_losses
        }, filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath)
        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_returns = checkpoint['episode_returns']
        self.episode_lengths = checkpoint['episode_lengths']
        self.policy_losses = checkpoint['policy_losses']
        print(f"📂 Model loaded from {filepath}")

# Test the REINFORCE agent
def test_reinforce_agent():
    """
    Test the REINFORCE agent to make sure all components work
    """
    print("=== TESTING REINFORCE AGENT ===")
    
    # Import our policy network
    from networks import PolicyNetwork
    
    # Create policy network and agent
    policy_net = PolicyNetwork(state_size=36, action_size=3, hidden_sizes=[64, 64])
    agent = REINFORCEAgent(policy_net, learning_rate=0.01, gamma=0.99)
    
    # Simulate a simple episode
    print("\n🎮 Simulating a test episode...")
    
    # Fake episode data
    test_states = [
        np.random.randn(36),  # Random state 1
        np.random.randn(36),  # Random state 2
        np.random.randn(36),  # Random state 3
    ]
    
    test_rewards = [0, 0, 2]  # No reward until end, then win 2 chips
    legal_actions = [0, 1, 2]  # All actions legal
    
    # Simulate agent actions
    for i, (state, reward) in enumerate(zip(test_states, test_rewards)):
        print(f"Step {i+1}:")
        action = agent.select_action(state, legal_actions)
        agent.store_reward(reward)
        print(f"  State shape: {state.shape}")
        print(f"  Action taken: {action}")
        print(f"  Reward: {reward}")
    
    # Update policy
    print("\n🧠 Updating policy...")
    loss = agent.update_policy()
    print(f"Policy loss: {loss:.6f}")
    
    # Check statistics
    stats = agent.get_training_stats()
    print(f"\nTraining stats: {stats}")
    
    print("\n✅ REINFORCE agent test completed!")

if __name__ == "__main__":
    test_reinforce_agent()
    
    print("\n" + "="*60)
    print("🎉 REINFORCE AGENT READY!")
    print("Next: We'll put everything together in a training loop")
    print("This will train our agent to play poker!")
    print("="*60)