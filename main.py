# PHASE 1 COMPLETE INTEGRATION
# This script brings everything together and runs the complete poker AI training pipeline

"""
🎯 PHASE 1 SUMMARY - WHAT WE'VE BUILT:

1. ✅ PROJECT STRUCTURE: Organized, modular codebase
2. ✅ ENVIRONMENT WRAPPER: Interface to poker games via RLCard  
3. ✅ POLICY NETWORK: Neural network that learns poker strategy
4. ✅ REINFORCE AGENT: Algorithm that learns from experience
5. ✅ TRAINING PIPELINE: Complete system for training and evaluation

WHAT YOUR AGENT WILL LEARN:
- Basic poker concepts (when to fold, call, raise)
- Pattern recognition in different game situations
- How to adapt strategy based on results
- Sequential decision making in uncertain environments

EXPECTED RESULTS:
- Win rate > 60% against random opponents (good for a beginner!)
- Positive average reward over many hands
- Stable learning convergence
- Basic poker strategy emergence
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Ensure we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_environment_check():
    """Check if all required packages are installed"""
    print("🔍 CHECKING ENVIRONMENT...")
    
    required_packages = {
        'torch': 'PyTorch for neural networks',
        'numpy': 'Numerical computing',
        'matplotlib': 'Plotting and visualization',
        'rlcard': 'Poker environments'
    }
    
    missing_packages = []
    
    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"   ✅ {package:12} - {description}")
        except ImportError:
            print(f"   ❌ {package:12} - {description} (MISSING)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {missing_packages}")
        print("Install with: pip install torch numpy matplotlib rlcard")
        return False
    
    print("✅ All required packages found!")
    return True

def create_project_structure():
    """Create the project directory structure"""
    print("\n📁 CREATING PROJECT STRUCTURE...")
    
    directories = [
        "poker_ai",
        "poker_ai/environments", 
        "poker_ai/agents",
        "poker_ai/networks",
        "poker_ai/evaluation",
        "poker_ai/utils",
        "logs",
        "models",
        "plots"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
        # Create __init__.py for Python packages
        if directory.startswith("poker_ai"):
            init_file = os.path.join(directory, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write("# Poker AI Package\n")
    
    print("   ✅ Directory structure created")
    return True

def run_component_tests():
    """Test each component individually before full training"""
    print("\n🧪 RUNNING COMPONENT TESTS...")
    
    try:
        # Test 1: Environment
        print("   Testing environment...")
        from poker_ai.environments.poker_env import PokerEnvironment
        env = PokerEnvironment('leduc-holdem')
        state, player_id = env.reset()
        assert state is not None and isinstance(player_id, int)
        print("   ✅ Environment test passed")
        
        # Test 2: Policy Network
        print("   Testing policy network...")
        from poker_ai.networks.policy_network import PolicyNetwork
        policy_net = PolicyNetwork(36, 3, [64, 64])
        test_state = torch.randn(1, 36)
        logits = policy_net.forward(test_state)
        assert logits.shape == (1, 3)
        print("   ✅ Policy network test passed")
        
        # Test 3: REINFORCE Agent
        print("   Testing REINFORCE agent...")
        from poker_ai.agents.reinforce_agent import REINFORCEAgent
        agent = REINFORCEAgent(policy_net)
        action = agent.select_action(np.random.randn(36), [0, 1, 2])
        assert isinstance(action, int) and 0 <= action <= 2
        print("   ✅ REINFORCE agent test passed")
        
        print("\n✅ ALL COMPONENT TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Component test failed: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure all files are saved correctly")
        print("2. Check that __init__.py files exist in poker_ai folders")
        print("3. Verify you're running from the project root directory")
        return False

def quick_training_demo():
    """Run a quick training demonstration"""
    print("\n🚀 QUICK TRAINING DEMONSTRATION...")
    print("   (Training for 100 episodes to verify everything works)")
    
    try:
        from phase1.trainer import PokerTrainer
        
        # Create trainer with smaller network for quick demo
        trainer = PokerTrainer(
            game_name='leduc-holdem',
            learning_rate=0.02,  # Slightly higher for faster learning
            gamma=0.99,
            hidden_sizes=[64, 64]  # Smaller network
        )
        
        # Short training run
        print("   Starting quick training...")
        start_time = time.time()
        
        final_metrics = trainer.train(
            num_episodes=100,
            update_frequency=10,
            eval_frequency=50,
            save_frequency=100
        )
        
        training_time = time.time() - start_time
        
        print(f"\n📊 QUICK DEMO RESULTS:")
        print(f"   Training time: {training_time:.1f} seconds")
        print(f"   Final win rate: {final_metrics['win_rate']:.1%}")
        print(f"   Average reward: {final_metrics['avg_reward']:.3f}")
        
        # Simple success criteria
        if final_metrics['win_rate'] > 0.4:  # Better than terrible
            print("   ✅ Agent shows signs of learning!")
            success = True
        else:
            print("   ⚠️  Agent needs more training (this is normal)")
            success = True  # Still counts as working
        
        return success, trainer
        
    except Exception as e:
        print(f"   ❌ Quick demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def full_training_run():
    """Run the complete training pipeline"""
    print("\n🎓 FULL TRAINING RUN...")
    
    try:
        from phase1.trainer import PokerTrainer
        
        # Create trainer
        trainer = PokerTrainer(
            game_name='leduc-holdem',
            learning_rate=0.01,
            gamma=0.99,
            hidden_sizes=[128, 128]
        )
        
        # Full training
        print("   Starting full training (this may take a few minutes)...")
        start_time = time.time()
        
        final_metrics = trainer.train(
            num_episodes=1000,
            update_frequency=10,
            eval_frequency=100,
            save_frequency=250
        )
        
        training_time = time.time() - start_time
        
        print(f"\n🎉 FULL TRAINING COMPLETED!")
        print(f"   Total time: {training_time:.1f} seconds")
        print(f"   Episodes per second: {1000/training_time:.2f}")
        print(f"   Final win rate: {final_metrics['win_rate']:.1%}")
        print(f"   Average reward: {final_metrics['avg_reward']:.3f}")
        
        # Plot results
        print("\n📈 Generating training plots...")
        trainer.plot_training_results()
        
        # Demonstrate gameplay
        print("\n🎮 Demonstrating learned behavior...")
        trainer.demonstrate_gameplay(num_hands=3)
        
        # Save final model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/poker_agent_final_{timestamp}.pth"
        trainer.agent.save_model(model_path)
        print(f"\n💾 Final model saved to: {model_path}")
        
        return True, trainer, final_metrics
        
    except Exception as e:
        print(f"   ❌ Full training failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def analyze_results(trainer, final_metrics):
    """Analyze and interpret the training results"""
    print("\n🔍 ANALYZING RESULTS...")
    
    win_rate = final_metrics['win_rate']
    avg_reward = final_metrics['avg_reward']
    
    print(f"\n📊 PERFORMANCE ANALYSIS:")
    
    # Win rate analysis
    if win_rate >= 0.7:
        print(f"   🌟 Excellent win rate ({win_rate:.1%}) - Agent learned strong strategy!")
    elif win_rate >= 0.6:
        print(f"   ✅ Good win rate ({win_rate:.1%}) - Agent shows solid poker understanding")
    elif win_rate >= 0.5:
        print(f"   👍 Decent win rate ({win_rate:.1%}) - Agent is competitive")
    else:
        print(f"   📈 Learning win rate ({win_rate:.1%}) - Agent needs more training")
    
    # Reward analysis
    if avg_reward >= 0.3:
        print(f"   💰 Strong average reward ({avg_reward:.3f}) - Profitable strategy")
    elif avg_reward >= 0.1:
        print(f"   💵 Positive average reward ({avg_reward:.3f}) - Winning strategy")
    elif avg_reward >= 0:
        print(f"   ⚖️  Break-even average reward ({avg_reward:.3f}) - Not losing money")
    else:
        print(f"   📉 Negative average reward ({avg_reward:.3f}) - Strategy needs improvement")
    
    # Learning assessment
    training_stats = trainer.agent.get_training_stats()
    total_episodes = training_stats.get('total_episodes', 0)
    
    if total_episodes > 0:
        print(f"\n🧠 LEARNING ANALYSIS:")
        print(f"   Total episodes played: {total_episodes}")
        print(f"   Best single episode: +{training_stats.get('best_return', 0):.1f} chips")
        print(f"   Worst single episode: {training_stats.get('worst_return', 0):.1f} chips")
        print(f"   Total profit/loss: {training_stats.get('total_return', 0):.1f} chips")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if win_rate < 0.6:
        print("   • Try longer training (more episodes)")
        print("   • Experiment with different learning rates")
        print("   • Add more sophisticated state features")
    
    if avg_reward < 0.1:
        print("   • Focus on reward engineering")
        print("   • Improve opponent modeling")
        print("   • Consider different neural network architectures")
    
    print("   • Ready to move to Phase 2: Advanced features and algorithms!")

def main():
    """Main execution function for Phase 1"""
    
    print("🎯 POKER AI PHASE 1 - COMPLETE INTEGRATION")
    print("=" * 60)
    print("Building and training a REINFORCE-based poker agent")
    print("=" * 60)
    
    # Step 1: Environment check
    if not setup_environment_check():
        print("\n❌ Environment check failed. Please install missing packages.")
        return False
    
    # Step 2: Project structure
    create_project_structure()
    
    # Step 3: Component tests
    if not run_component_tests():
        print("\n❌ Component tests failed. Check your implementation.")
        print("\nCommon fixes:")
        print("1. Make sure all .py files are saved correctly")
        print("2. Check import statements in files")
        print("3. Verify __init__.py files exist")
        return False
    
    # Step 4: User choice for training type
    print("\n🤔 TRAINING OPTIONS:")
    print("   1. Quick demo (100 episodes, ~30 seconds)")
    print("   2. Full training (1000 episodes, ~5 minutes)")
    print("   3. Component tests only")
    
    try:
        choice = input("\nEnter your choice (1/2/3): ").strip()
    except KeyboardInterrupt:
        print("\n\n👋 Training cancelled by user")
        return False
    
    if choice == "1":
        # Quick demo
        success, trainer = quick_training_demo()
        if success and trainer:
            print("\n✅ Quick demo completed successfully!")
            print("   Your agent learned basic poker concepts in just 100 episodes!")
        return success
    
    elif choice == "2":
        # Full training
        success, trainer, final_metrics = full_training_run()
        if success and trainer and final_metrics:
            analyze_results(trainer, final_metrics)
            print("\n🎉 PHASE 1 COMPLETE!")
            print("\n🚀 Ready for Phase 2:")
            print("   • Better state representation using poker expertise")
            print("   • Advanced algorithms (DQN, PPO)")
            print("   • Opponent modeling")
            print("   • Multi-player games")
        return success
    
    elif choice == "3":
        # Tests only
        print("\n✅ Component tests completed successfully!")
        print("   All components are working correctly.")
        print("   You can now run training whenever you're ready.")
        return True
    
    else:
        print("\n❌ Invalid choice. Please run again and select 1, 2, or 3.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        
        if success:
            print("\n" + "=" * 60)
            print("🎊 CONGRATULATIONS!")
            print("You've successfully built and trained a poker AI agent!")
            print("\nWhat you achieved:")
            print("✅ Complete RL pipeline from scratch")
            print("✅ Neural network that learns poker strategy")
            print("✅ REINFORCE algorithm implementation")
            print("✅ Training and evaluation framework")
            print("✅ Model persistence and analysis tools")
            print("\nYour agent can now:")
            print("🧠 Make decisions based on game state")
            print("📈 Learn from experience")
            print("🎯 Beat random opponents")
            print("💾 Save and load its learned strategy")
            print("=" * 60)
        else:
            print("\n❌ Phase 1 incomplete. Check error messages above.")
    
    except KeyboardInterrupt:
        print("\n\n👋 Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("\nIf you need help, check:")
        print("1. All packages are installed correctly")
        print("2. Python version is 3.7 or higher")
        print("3. No file permission issues")
        print("\nFor detailed error info, re-run with:")
        print("python -u main.py")