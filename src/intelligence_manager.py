import numpy as np
import os

class IntelligenceManager:
    """Handles saving and loading the agent's neural pathways."""
    
    @staticmethod
    def save_intelligence(guide, filename="city_ai_weights.npz"):
        """Saves weight matrices (W1, W2) to a compressed file."""
        # We ensure CAPITAL W1 and W2 to match brain.py exactly
        np.savez(filename, W1=guide.W1, W2=guide.W2)
        print(f"✅ Intelligence saved to {filename}")

    @staticmethod
    def load_intelligence(guide, filename="city_ai_weights.npz"):
        """Loads weights from a file if it exists."""
        if os.path.exists(filename):
            data = np.load(filename)
            # We map the saved data back to the guide's Capital W variables
            guide.W1 = data['W1']
            guide.W2 = data['W2']
            print(f"🧠 Intelligence loaded from {filename}")
            return True
        print("⚠️ No previous intelligence found. Starting fresh.")
        return False