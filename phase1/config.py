# STEP 1 - Configuration and Project Setup
import os
import logging
from datetime import datetime

class Config:
    """Configuration class for our Poker AI"""
    
    # ENVIRONMENT SETTINGS
    GAME_NAME = 'leduc-holdem'
    NUM_PLAYERS = 2
    
    # TRAINING HYPERPARAMETERS
    LEARNING_RATE = 0.01
    GAMMA = 0.99
    
    # NETWORK ARCHITECTURE
    HIDDEN_SIZE = 128
    
    # TRAINING SETTINGS
    NUM_EPISODES = 1000
    UPDATE_FREQUENCY = 10
    EVAL_FREQUENCY = 100
    
    # LOGGING AND SAVING
    LOG_DIR = "logs/"
    MODEL_DIR = "models/"
    SAVE_FREQUENCY = 500

def setup_project_structure():
    """Create the folder structure for our poker AI project"""
    folders = [
        "poker_ai/environments",
        "poker_ai/agents", 
        "poker_ai/networks",
        "poker_ai/evaluation",
        "poker_ai/utils",
        Config.LOG_DIR,
        Config.MODEL_DIR
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        
        if folder.startswith("poker_ai/"):
            init_file = os.path.join(folder, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write("# Poker AI Package\n")
    
    print("✅ Project structure created successfully!")
    return folders

def setup_logger():
    """Set up logging to track training progress"""
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    
    log_filename = f"{Config.LOG_DIR}/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 Poker AI training session started")
    logger.info(f"Configuration: {Config.__dict__}")
    
    return logger

if __name__ == "__main__":
    print("=== PHASE 1 STEP 1: PROJECT SETUP ===")
    setup_project_structure()
    logger = setup_logger()
    logger.info("Project setup completed successfully")
    print("✅ Configuration and project structure ready!")