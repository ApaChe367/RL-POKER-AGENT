# PHASE 1 - STEP 2: ENVIRONMENT WRAPPER
# This handles the interface between our RL agent and the poker game

"""
ENVIRONMENT WRAPPER EXPLANATION:

The environment wrapper is like a translator between:
1. The poker game (RLCard) - which speaks "poker language"
2. Our RL agent - which speaks "RL language"

Key responsibilities:
- Convert poker states to neural network inputs
- Handle action translation
- Manage game flow
- Provide consistent interface for different poker variants
"""

import numpy as np
import rlcard
from rlcard.agents import RandomAgent
from typing import Tuple, Dict, List, Any

class PokerEnvironment:
    """
    Environment wrapper for poker games using RLCard
    
    This class standardizes the interface so we can easily:
    - Switch between different poker variants
    - Change opponents 
    - Modify state representations
    - Add logging and debugging
    """
    
    def __init__(self, game_name: str = 'leduc-holdem', num_players: int = 2):
        """
        Initialize the poker environment
        
        Args:
            game_name: Which poker variant to play
            num_players: Number of players in the game
        """
        self.game_name = game_name
        self.num_players = num_players
        
        # Create the RLCard environment
        self.env = rlcard.make(game_name)
        
        # Store important environment info
        self.num_actions = self.env.num_actions
        self.state_shape = self.env.state_shape[0]  # State dimensions
        
        # Initialize opponent agents (we'll start with random opponents)
        self.opponents = self._create_opponents()
        
        print(f"🎮 Environment created: {game_name}")
        print(f"   Players: {num_players}")
        print(f"   Actions: {self.num_actions}")
        print(f"   State shape: {self.state_shape}")
        
        # Let's understand what these actions mean
        self._explain_actions()
    
    def _create_opponents(self) -> List[Any]:
        """
        Create opponent agents
        
        For now, we'll use random opponents
        Later we can add different opponent types
        """
        opponents = []
        for i in range(1, self.num_players):  # Skip player 0 (our agent)
            opponent = RandomAgent(num_actions=self.num_actions)
            opponents.append(opponent)
        
        print(f"🤖 Created {len(opponents)} random opponents")
        return opponents
    
    def _explain_actions(self):
        """
        Explain what each action means in poker terms
        This helps us understand what our agent is learning
        """
        print(f"\n📋 Action meanings for {self.game_name}:")
        
        if self.game_name == 'leduc-holdem':
            action_meanings = {
                0: "CALL/CHECK - Match current bet or check if no bet",
                1: "RAISE/BET - Increase the bet (usually by 2 chips)",
                2: "FOLD - Give up and lose any chips already invested"
            }
        else:
            # Generic meanings for other poker variants
            action_meanings = {
                0: "CALL/CHECK",
                1: "RAISE/BET", 
                2: "FOLD"
            }
        
        for action_id, meaning in action_meanings.items():
            if action_id < self.num_actions:
                print(f"   Action {action_id}: {meaning}")
    
    def reset(self) -> Tuple[np.ndarray, int]:
        """
        Start a new poker hand
        
        Returns:
            state: The initial game state for our agent
            player_id: Which player our agent is (usually 0)
        """
        # Reset the game environment
        state, player_id = self.env.reset()
        
        # For now, return the raw state - we'll process it later
        processed_state = self._process_state(state)
        
        return processed_state, player_id
    
    def step(self, action: int) -> Tuple[np.ndarray, int, bool, Dict]:
        """
        Execute one action in the game
        
        Args:
            action: The action our agent wants to take
            
        Returns:
            next_state: Game state after the action
            player_id: Which player acts next
            done: Whether the hand is finished
            info: Additional information
        """
        # In poker, multiple players take turns
        # We need to handle opponent actions too
        
        current_player = self.env.get_player_id()
        
        if current_player == 0:
            # Our agent's turn
            next_state, next_player_id = self.env.step(action)
        else:
            # Opponent's turn - use their strategy
            opponent_idx = current_player - 1
            opponent_action = self.opponents[opponent_idx].use_raw(
                self.env.get_state(current_player)
            )
            next_state, next_player_id = self.env.step(opponent_action)
        
        # Check if game is finished
        done = self.env.is_over()
        
        # Process the state for our agent
        processed_state = self._process_state(next_state) if not done else None
        
        # Additional info (useful for debugging)
        info = {
            'current_player': current_player,
            'action_taken': action if current_player == 0 else opponent_action,
            'legal_actions': next_state['legal_actions'] if not done else []
        }
        
        return processed_state, next_player_id, done, info
    
    def _process_state(self, raw_state: Dict) -> np.ndarray:
        """
        Convert raw game state to neural network input
        
        This is where we'll later add poker-specific feature engineering
        For now, we'll use the raw observation
        
        YOUR POKER EXPERTISE WILL BE CRUCIAL HERE!
        Later we'll add features like:
        - Hand strength estimation
        - Pot odds calculation  
        - Betting pattern analysis
        - Position information
        """
        
        # For now, just return the raw observation
        # RLCard already provides a numerical representation
        return raw_state['obs']
    
    def get_legal_actions(self) -> List[int]:
        """
        Get actions that are legal in the current state
        
        In poker, not all actions are always available:
        - Can't fold if there's no bet to call
        - Can't raise if already all-in
        - etc.
        """
        if self.env.is_over():
            return []
        
        current_state = self.env.get_state(self.env.get_player_id())
        return current_state['legal_actions']
    
    def get_payoffs(self) -> List[float]:
        """
        Get the final rewards for all players
        
        In poker:
        - Positive reward = won chips
        - Negative reward = lost chips  
        - Zero-sum game: one player's gain = others' loss
        """
        return self.env.get_payoffs()
    
    def render(self):
        """
        Display the current game state
        Useful for debugging and understanding what's happening
        """
        print("\n" + "="*50)
        print("CURRENT GAME STATE")
        print("="*50)
        
        if not self.env.is_over():
            current_player = self.env.get_player_id()
            state = self.env.get_state(current_player)
            
            print(f"Current player: {current_player}")
            print(f"Legal actions: {state['legal_actions']}")
            print(f"Raw observation: {state['obs']}")
        else:
            payoffs = self.get_payoffs()
            print("HAND FINISHED!")
            print(f"Final payoffs: {payoffs}")
            winner = np.argmax(payoffs)
            print(f"Winner: Player {winner} (+{payoffs[winner]} chips)")

# Let's test our environment wrapper
def test_environment():
    """
    Test the environment wrapper to make sure it works correctly
    """
    print("=== TESTING ENVIRONMENT WRAPPER ===")
    
    # Create environment
    env = PokerEnvironment('leduc-holdem', num_players=2)
    
    # Play one complete hand to test all functionality
    print("\n🎯 Playing one test hand...")
    
    state, player_id = env.reset()
    print(f"Initial state shape: {state.shape}")
    print(f"Initial player: {player_id}")
    
    step_count = 0
    while True:
        # Show current state
        env.render()
        
        # Get legal actions
        legal_actions = env.get_legal_actions()
        print(f"Legal actions: {legal_actions}")
        
        if not legal_actions:  # Game over
            break
            
        # Take a random action (our agent will do better later!)
        action = np.random.choice(legal_actions)
        print(f"Taking action: {action}")
        
        # Execute action
        next_state, next_player_id, done, info = env.step(action)
        
        step_count += 1
        if step_count > 20:  # Safety check to avoid infinite loops
            print("⚠️  Too many steps, breaking...")
            break
            
        if done:
            env.render()  # Show final state
            break
        
        state = next_state
        player_id = next_player_id
    
    print(f"✅ Test completed in {step_count} steps")

if __name__ == "__main__":
    test_environment()
    
    print("\n" + "="*60)
    print("🎉 ENVIRONMENT WRAPPER READY!")
    print("Next: We'll implement the neural network for our policy")
    print("="*60)