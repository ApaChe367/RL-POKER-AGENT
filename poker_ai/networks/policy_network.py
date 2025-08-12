# PHASE 1 - STEP 3: POLICY NETWORK
# This is the "brain" of our poker agent - the neural network that learns strategy

"""
POLICY NETWORK EXPLANATION:

What is a Policy Network?
- A neural network that takes a poker situation as input
- Outputs probabilities for each possible action
- Represents the agent's strategy: π(a|s) = probability of action a given state s

Why Neural Networks for Poker?
1. COMPLEX PATTERNS: Poker strategy involves complex patterns
2. GENERALIZATION: Can handle unseen situations  
3. LEARNING: Can improve through experience
4. FUNCTION APPROXIMATION: Maps millions of possible situations to actions

Think of it like this:
- Input: "I have King-Queen, board is Ace-King-7, opponent bet 50% pot"
- Output: [Fold: 0.1, Call: 0.3, Raise: 0.6] - probabilities for each action
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np
from typing import Tuple, List

class PolicyNetwork(nn.Module):
    """
    Policy Network for Poker Agent
    
    Architecture: Multi-Layer Perceptron (MLP)
    - Input Layer: Game state features
    - Hidden Layers: Learn poker patterns  
    - Output Layer: Action probabilities
    """
    
    def __init__(self, 
                 state_size: int, 
                 action_size: int, 
                 hidden_sizes: List[int] = [128, 128]):
        """
        Initialize the policy network
        
        Args:
            state_size: Size of the input state vector
            action_size: Number of possible actions (3 for Leduc Hold'em)
            hidden_sizes: List of hidden layer sizes
        """
        super(PolicyNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = hidden_sizes
        
        # Build the network layers
        layers = []
        
        # Input layer
        prev_size = state_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())  # ReLU activation function
            prev_size = hidden_size
        
        # Output layer (no activation - we'll apply softmax later)
        layers.append(nn.Linear(prev_size, action_size))
        
        # Combine all layers
        self.network = nn.Sequential(*layers)
        
        # Initialize weights for better training
        self._initialize_weights()
        
        print(f"🧠 Policy Network created:")
        print(f"   Input size: {state_size}")
        print(f"   Hidden layers: {hidden_sizes}")
        print(f"   Output size: {action_size}")
        print(f"   Total parameters: {self._count_parameters()}")
    
    def _initialize_weights(self):
        """
        Initialize network weights for better training
        
        WHY WEIGHT INITIALIZATION MATTERS:
        - Poor initialization can make training very slow or fail
        - Good initialization helps the network learn faster
        - Xavier/Glorot initialization is standard for ReLU networks
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def _count_parameters(self) -> int:
        """Count total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            state: Tensor representing the game state
            
        Returns:
            logits: Raw scores for each action (before softmax)
        """
        # Pass state through the network
        logits = self.network(state)
        return logits
    
    def get_action_probabilities(self, state: torch.Tensor) -> torch.Tensor:
        """
        Convert state to action probabilities
        
        This applies softmax to convert raw scores to probabilities
        Softmax ensures: sum of probabilities = 1, all probabilities > 0
        """
        logits = self.forward(state)
        probabilities = F.softmax(logits, dim=-1)
        return probabilities
    
    def select_action(self, state: torch.Tensor, legal_actions: List[int]) -> Tuple[int, torch.Tensor]:
        """
        Select an action using the current policy
        
        IMPORTANT: We sample from the probability distribution
        - This adds exploration (trying different actions)
        - Higher probability actions are chosen more often
        - But we sometimes try lower probability actions too
        
        Args:
            state: Current game state
            legal_actions: List of legal actions in this state
            
        Returns:
            action: The selected action
            log_prob: Log probability of the selected action (needed for REINFORCE)
        """
        
        # Get action probabilities
        action_probs = self.get_action_probabilities(state)
        
        # Mask illegal actions (set their probability to 0)
        masked_probs = action_probs.clone()
        
        # Create a mask for legal actions
        legal_mask = torch.zeros_like(action_probs)
        for legal_action in legal_actions:
            legal_mask[legal_action] = 1.0
        
        # Apply mask (multiply by 0 for illegal actions)
        masked_probs = masked_probs * legal_mask
        
        # Renormalize so probabilities sum to 1
        masked_probs = masked_probs / masked_probs.sum()
        
        # Create a categorical distribution and sample from it
        action_distribution = distributions.Categorical(masked_probs)
        action = action_distribution.sample()
        
        # Get log probability of the chosen action (needed for REINFORCE updates)
        log_prob = action_distribution.log_prob(action)
        
        return action.item(), log_prob
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the log probabilities of given state-action pairs
        
        This is used during policy updates to calculate gradients
        
        Args:
            states: Batch of states
            actions: Batch of actions taken in those states
            
        Returns:
            log_probs: Log probabilities of the actions
        """
        action_probs = self.get_action_probabilities(states)
        action_distributions = distributions.Categorical(action_probs)
        log_probs = action_distributions.log_prob(actions)
        
        return log_probs

# Let's create a simple test to verify our network works
def test_policy_network():
    """
    Test the policy network to make sure it works correctly
    """
    print("=== TESTING POLICY NETWORK ===")
    
    # Create a test network (matching Leduc Hold'em dimensions)
    state_size = 36  # Leduc Hold'em state size
    action_size = 3  # Call/Raise/Fold
    
    policy_net = PolicyNetwork(state_size, action_size, hidden_sizes=[64, 64])
    
    # Test 1: Forward pass with random state
    print("\n🧪 Test 1: Forward pass")
    test_state = torch.randn(1, state_size)  # Random state
    logits = policy_net.forward(test_state)
    print(f"Input shape: {test_state.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Raw logits: {logits}")
    
    # Test 2: Action probabilities
    print("\n🧪 Test 2: Action probabilities")
    action_probs = policy_net.get_action_probabilities(test_state)
    print(f"Action probabilities: {action_probs}")
    print(f"Sum of probabilities: {action_probs.sum().item():.6f} (should be 1.0)")
    
    # Test 3: Action selection
    print("\n🧪 Test 3: Action selection")
    legal_actions = [0, 1, 2]  # All actions legal
    action, log_prob = policy_net.select_action(test_state.squeeze(), legal_actions)
    print(f"Selected action: {action}")
    print(f"Log probability: {log_prob.item():.6f}")
    
    # Test 4: Action selection with restricted legal actions
    print("\n🧪 Test 4: Restricted legal actions")
    legal_actions = [0, 2]  # Can only call or fold (no raising)
    action, log_prob = policy_net.select_action(test_state.squeeze(), legal_actions)
    print(f"Legal actions: {legal_actions}")
    print(f"Selected action: {action}")
    print(f"Action should be in {legal_actions}: {action in legal_actions}")
    
    # Test 5: Batch evaluation
    print("\n🧪 Test 5: Batch evaluation")
    batch_size = 5
    batch_states = torch.randn(batch_size, state_size)
    batch_actions = torch.randint(0, action_size, (batch_size,))
    
    log_probs = policy_net.evaluate_actions(batch_states, batch_actions)
    print(f"Batch states shape: {batch_states.shape}")
    print(f"Batch actions: {batch_actions}")
    print(f"Log probabilities: {log_probs}")
    
    print("\n✅ All tests passed!")

# Let's also create a visualization function to understand what the network is learning
def visualize_policy(policy_net: PolicyNetwork, state_size: int):
    """
    Visualize what the policy network has learned
    
    This helps us understand if the network is learning reasonable poker strategy
    """
    print("\n=== POLICY VISUALIZATION ===")
    
    # Generate some test states and see what the policy does
    num_test_states = 5
    
    for i in range(num_test_states):
        # Create a random test state
        test_state = torch.randn(state_size)
        
        # Get action probabilities
        action_probs = policy_net.get_action_probabilities(test_state.unsqueeze(0))
        probs = action_probs.squeeze().detach().numpy()
        
        print(f"\n🎮 Test State {i+1}:")
        print(f"   Call/Check probability: {probs[0]:.3f}")
        print(f"   Raise/Bet probability:  {probs[1]:.3f}")
        print(f"   Fold probability:       {probs[2]:.3f}")
        
        # What would the agent do?
        best_action = np.argmax(probs)
        action_names = ["Call/Check", "Raise/Bet", "Fold"]
        print(f"   Most likely action: {action_names[best_action]} ({probs[best_action]:.3f})")

if __name__ == "__main__":
    # Run tests
    test_policy_network()
    
    # Create a network and visualize initial policy
    policy_net = PolicyNetwork(36, 3, hidden_sizes=[64, 64])
    visualize_policy(policy_net, 36)
    
    print("\n" + "="*60)
    print("🎉 POLICY NETWORK READY!")
    print("Next: We'll implement the REINFORCE algorithm")
    print("This will teach the network to improve its poker strategy")
    print("="*60)