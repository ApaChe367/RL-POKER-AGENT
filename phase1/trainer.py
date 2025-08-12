# PHASE 1 - STEP 5: COMPLETE TRAINING LOOP
# This brings everything together and trains our poker agent

"""
TRAINING LOOP EXPLANATION:

This is where the magic happens! The training loop:

1. EPISODE LOOP: Play many poker hands
2. EXPERIENCE COLLECTION: Agent takes actions, receives rewards
3. POLICY UPDATES: Learn from experience to improve strategy
4. EVALUATION: Monitor progress and performance
5. LOGGING: Track training progress

The full cycle:
Play Hand → Collect Experience → Update Policy → Evaluate → Repeat
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from typing import Dict, List, Tuple
import logging

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our components
try:
    from poker_ai.environments.poker_env import PokerEnvironment
    from poker_ai.networks.policy_network import PolicyNetwork
    from poker_ai.agents.reinforce_agent import REINFORCEAgent
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    print("And that all __init__.py files are created in poker_ai folders")
    sys.exit(1)

class PokerTrainer:
    """
    Complete training system for poker agents
    
    This class orchestrates the entire training process:
    - Environment management
    - Agent training
    - Performance evaluation
    - Progress logging
    - Model saving/loading
    """
    
    def __init__(self, 
                 game_name: str = 'leduc-holdem',
                 learning_rate: float = 0.01,
                 gamma: float = 0.99,
                 hidden_sizes: List[int] = [128, 128]):
        """
        Initialize the poker trainer
        
        Args:
            game_name: Which poker variant to train on
            learning_rate: How fast the agent learns
            gamma: Discount factor for future rewards
            hidden_sizes: Neural network architecture
        """
        
        print("🚀 INITIALIZING POKER TRAINER")
        print("=" * 50)
        
        # Create environment
        self.env = PokerEnvironment(game_name, num_players=2)
        
        # Create policy network
        state_size = self.env.state_shape[0]
        action_size = self.env.num_actions
        
        self.policy_network = PolicyNetwork(
            state_size=state_size,
            action_size=action_size, 
            hidden_sizes=hidden_sizes
        )
        
        # Create REINFORCE agent
        self.agent = REINFORCEAgent(
            policy_network=self.policy_network,
            learning_rate=learning_rate,
            gamma=gamma
        )
        
        # Training configuration
        self.game_name = game_name
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'win_rates': [],
            'evaluation_scores': []
        }
        
        print(f"✅ Trainer initialized for {game_name}")
        print(f"   State size: {state_size}")
        print(f"   Action size: {action_size}")
        print(f"   Network parameters: {self._count_parameters()}")
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.policy_network.parameters() if p.requires_grad)
    
    def play_episode(self) -> Tuple[float, int]:
        """
        Play one complete poker hand
        
        This is the core of our training - one episode = one poker hand
        
        Returns:
            episode_reward: Total reward for our agent
            episode_length: Number of actions taken
        """
        
        # Reset environment for new hand
        state, player_id = self.env.reset()
        
        episode_reward = 0
        episode_length = 0
        
        # Play until hand is complete
        while True:
            # Get legal actions
            legal_actions = self.env.get_legal_actions()
            
            if not legal_actions:  # Hand is over
                break
            
            # Check whose turn it is
            current_player = self.env.env.get_player_id()
            
            if current_player == 0:  # Our agent's turn
                # Agent selects action
                action = self.agent.select_action(state, legal_actions)
                
                # Execute action
                next_state, next_player_id, done, info = self.env.step(action)
                
                episode_length += 1
                
                if done:
                    # Hand finished, get final rewards
                    payoffs = self.env.get_payoffs()
                    final_reward = payoffs[0]  # Our agent's payoff
                    episode_reward = final_reward
                    
                    # Store final reward for learning
                    self.agent.store_reward(final_reward)
                    break
                else:
                    # Hand continues, no immediate reward
                    self.agent.store_reward(0)
                    state = next_state
            
            else:
                # Opponent's turn - they act automatically
                next_state, next_player_id, done, info = self.env.step(0)  # Dummy action
                
                if done:
                    # Hand finished
                    payoffs = self.env.get_payoffs()
                    final_reward = payoffs[0]
                    episode_reward = final_reward
                    break
                else:
                    state = next_state
        
        return episode_reward, episode_length
    
    def evaluate_agent(self, num_episodes: int = 100) -> Dict[str, float]:
        """
        Evaluate agent performance without learning
        
        This gives us an unbiased estimate of how good our agent is
        
        Args:
            num_episodes: Number of hands to evaluate over
            
        Returns:
            evaluation_metrics: Win rate, average reward, etc.
        """
        
        print(f"\n📊 Evaluating agent over {num_episodes} episodes...")
        
        # Temporarily store agent's learning data
        saved_log_probs = self.agent.episode_log_probs.copy()
        saved_rewards = self.agent.episode_rewards.copy()
        saved_states = self.agent.episode_states.copy()
        saved_actions = self.agent.episode_actions.copy()
        
        # Clear learning data for evaluation
        self.agent.episode_log_probs = []
        self.agent.episode_rewards = []
        self.agent.episode_states = []
        self.agent.episode_actions = []
        
        # Evaluation metrics
        total_reward = 0
        wins = 0
        total_length = 0
        
        for episode in range(num_episodes):
            reward, length = self.play_episode()
            
            total_reward += reward
            total_length += length
            if reward > 0:
                wins += 1
            
            # Clear learning data after each evaluation episode
            self.agent.episode_log_probs = []
            self.agent.episode_rewards = []
            self.agent.episode_states = []
            self.agent.episode_actions = []
        
        # Restore agent's learning data
        self.agent.episode_log_probs = saved_log_probs
        self.agent.episode_rewards = saved_rewards
        self.agent.episode_states = saved_states
        self.agent.episode_actions = saved_actions
        
        # Calculate metrics
        win_rate = wins / num_episodes
        avg_reward = total_reward / num_episodes
        avg_length = total_length / num_episodes
        
        metrics = {
            'win_rate': win_rate,
            'avg_reward': avg_reward,
            'avg_length': avg_length,
            'total_reward': total_reward,
            'total_wins': wins
        }
        
        print(f"   Win Rate: {win_rate:.1%}")
        print(f"   Average Reward: {avg_reward:.3f}")
        print(f"   Average Length: {avg_length:.1f}")
        
        return metrics
    
    def train(self, 
              num_episodes: int = 1000,
              update_frequency: int = 10,
              eval_frequency: int = 100,
              save_frequency: int = 500):
        """
        Main training loop
        
        This is where our agent learns to play poker!
        
        Args:
            num_episodes: Total number of poker hands to play
            update_frequency: Update policy every N episodes
            eval_frequency: Evaluate performance every N episodes
            save_frequency: Save model every N episodes
        """
        
        print(f"\n🎓 STARTING TRAINING")
        print("=" * 50)
        print(f"Total episodes: {num_episodes}")
        print(f"Update frequency: {update_frequency}")
        print(f"Evaluation frequency: {eval_frequency}")
        print()
        
        start_time = time.time()
        episodes_since_update = 0
        
        for episode in range(1, num_episodes + 1):
            # Play one episode
            reward, length = self.play_episode()
            
            # Store training statistics
            self.training_stats['episode_rewards'].append(reward)
            self.training_stats['episode_lengths'].append(length)
            
            episodes_since_update += 1
            
            # Update policy
            if episodes_since_update >= update_frequency:
                loss = self.agent.update_policy()
                episodes_since_update = 0
                
                # Print training progress
                recent_rewards = self.training_stats['episode_rewards'][-update_frequency:]
                avg_reward = np.mean(recent_rewards)
                win_rate = np.mean([r > 0 for r in recent_rewards])
                
                print(f"Episode {episode:4d} | "
                      f"Avg Reward: {avg_reward:6.3f} | "
                      f"Win Rate: {win_rate:5.1%} | "
                      f"Loss: {loss:8.3f}")
            
            # Periodic evaluation
            if episode % eval_frequency == 0:
                eval_metrics = self.evaluate_agent(num_episodes=50)
                self.training_stats['evaluation_scores'].append(eval_metrics)
                
                print(f"\n📈 EVALUATION at Episode {episode}:")
                print(f"   Win Rate: {eval_metrics['win_rate']:.1%}")
                print(f"   Avg Reward: {eval_metrics['avg_reward']:.3f}")
                print()
            
            # Save model
            if episode % save_frequency == 0:
                model_path = f"models/poker_agent_episode_{episode}.pth"
                os.makedirs("models", exist_ok=True)
                self.agent.save_model(model_path)
        
        # Training completed
        total_time = time.time() - start_time
        print(f"\n🎉 TRAINING COMPLETED!")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Episodes per second: {num_episodes/total_time:.2f}")
        
        # Final evaluation
        print(f"\n📊 FINAL EVALUATION:")
        final_metrics = self.evaluate_agent(num_episodes=200)
        
        return final_metrics
    
    def plot_training_results(self):
        """Plot comprehensive training results"""
        
        if len(self.training_stats['episode_rewards']) < 10:
            print("Not enough training data to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot 1: Episode rewards
        axes[0, 0].plot(self.training_stats['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # Plot 2: Running average reward
        window = 50
        if len(self.training_stats['episode_rewards']) > window:
            running_avg = []
            for i in range(len(self.training_stats['episode_rewards'])):
                start_idx = max(0, i - window + 1)
                avg = np.mean(self.training_stats['episode_rewards'][start_idx:i+1])
                running_avg.append(avg)
            
            axes[0, 1].plot(running_avg)
            axes[0, 1].set_title(f'Running Average Reward (window={window})')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Average Reward')
            axes[0, 1].grid(True)
        
        # Plot 3: Episode lengths
        axes[0, 2].plot(self.training_stats['episode_lengths'])
        axes[0, 2].set_title('Episode Lengths')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Actions per Episode')
        axes[0, 2].grid(True)
        
        # Plot 4: Win rate over time
        if len(self.training_stats['episode_rewards']) > 20:
            window = 20
            win_rates = []
            for i in range(window, len(self.training_stats['episode_rewards'])):
                recent_rewards = self.training_stats['episode_rewards'][i-window:i]
                win_rate = np.mean([r > 0 for r in recent_rewards])
                win_rates.append(win_rate)
            
            axes[1, 0].plot(range(window, len(self.training_stats['episode_rewards'])), win_rates)
            axes[1, 0].set_title(f'Win Rate Over Time (window={window})')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Win Rate')
            axes[1, 0].grid(True)
        
        # Plot 5: Evaluation scores
        if self.training_stats['evaluation_scores']:
            eval_episodes = []
            eval_win_rates = []
            eval_avg_rewards = []
            
            for i, score in enumerate(self.training_stats['evaluation_scores']):
                eval_episodes.append((i + 1) * 100)  # Assuming eval every 100 episodes
                eval_win_rates.append(score['win_rate'])
                eval_avg_rewards.append(score['avg_reward'])
            
            axes[1, 1].plot(eval_episodes, eval_win_rates, 'o-')
            axes[1, 1].set_title('Evaluation Win Rates')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Win Rate')
            axes[1, 1].grid(True)
            
            axes[1, 2].plot(eval_episodes, eval_avg_rewards, 'o-')
            axes[1, 2].set_title('Evaluation Average Rewards')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Average Reward')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def demonstrate_gameplay(self, num_hands: int = 3):
        """Show detailed gameplay to understand what the agent learned"""
        
        print(f"\n🎮 DEMONSTRATING AGENT GAMEPLAY")
        print("=" * 50)
        
        for hand in range(num_hands):
            print(f"\n🃏 HAND {hand + 1}:")
            print("-" * 30)
            
            state, player_id = self.env.reset()
            step = 0
            
            while True:
                # Show current state
                legal_actions = self.env.get_legal_actions()
                
                if not legal_actions:
                    break
                
                current_player = self.env.env.get_player_id()
                
                if current_player == 0:  # Our agent
                    # Get action probabilities to see agent's "thinking"
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action_probs = self.policy_network.get_action_probabilities(state_tensor)
                    probs = action_probs.squeeze().detach().numpy()
                    
                    action_names = ["Call/Check", "Raise/Bet", "Fold"]
                    
                    print(f"Step {step + 1} - Agent's turn:")
                    print(f"  Legal actions: {legal_actions}")
                    print(f"  Action probabilities:")
                    for i, prob in enumerate(probs):
                        if i in legal_actions:
                            print(f"    {action_names[i]}: {prob:.3f}")
                    
                    # Agent selects action (without storing for learning)
                    action, _ = self.policy_network.select_action(state_tensor.squeeze(), legal_actions)
                    print(f"  Agent chooses: {action_names[action]}")
                    
                    # Execute action
                    next_state, next_player_id, done, info = self.env.step(action)
                    
                    if done:
                        payoffs = self.env.get_payoffs()
                        print(f"  Hand result: Agent gets {payoffs[0]} chips")
                        break
                    else:
                        state = next_state
                        step += 1
                else:
                    # Opponent's turn
                    print(f"Step {step + 1} - Opponent's turn")
                    next_state, next_player_id, done, info = self.env.step(0)
                    
                    if done:
                        payoffs = self.env.get_payoffs()
                        print(f"  Hand result: Agent gets {payoffs[0]} chips")
                        break
                    else:
                        state = next_state
                        step += 1

# Test function
def test_trainer():
    """Test the trainer to make sure it works"""
    print("=== TESTING TRAINER ===")
    
    try:
        # Create trainer
        trainer = PokerTrainer(
            game_name='leduc-holdem',
            learning_rate=0.02,
            gamma=0.99,
            hidden_sizes=[64, 64]
        )
        
        # Quick training test
        print("\n🧪 Running quick training test (10 episodes)...")
        final_metrics = trainer.train(
            num_episodes=10,
            update_frequency=5,
            eval_frequency=10,
            save_frequency=10
        )
        
        print(f"\n✅ Trainer test completed!")
        print(f"   Win rate: {final_metrics['win_rate']:.1%}")
        print(f"   Avg reward: {final_metrics['avg_reward']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Trainer test failed: {e}")
        return False

if __name__ == "__main__":
    # Test the trainer
    success = test_trainer()
    
    if success:
        print("\n🎉 TRAINER READY!")
        print("You can now run the main training pipeline!")
    else:
        print("\n❌ Trainer test failed. Check error messages above.")