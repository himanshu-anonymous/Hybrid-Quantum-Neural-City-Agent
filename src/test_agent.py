from topology import NetworkTopology
from quantum_engine import QuantumExplorer
from brain import NeuralGuide
from visualizer import CityVisualizer
from intelligence_manager import IntelligenceManager
import numpy as np
import matplotlib.pyplot as plt

def run_test_deployment():
    """Runs the agent in 100% exploitation mode using saved intelligence."""
    print("Loading Trained Intelligence for Deployment...")
    
    world = NetworkTopology()
    explorer = QuantumExplorer()
    # Ensure hidden_size matches the 12 used in your latest main.py training
    guide = NeuralGuide(input_size=3, hidden_size=12) 
    
    # Load the intelligence saved by main.py
    if not IntelligenceManager.load_intelligence(guide):
        print("Error: No trained weights found. Please run main.py first.")
        return

    viz = CityVisualizer(world)
    test_runs = 5 

    for run in range(test_runs):
        world.update_traffic() # Generate a new dynamic traffic scenario
        curr = world.start_pos
        path = [curr]
        total_reward = 0
        
        print(f"\n Running Deployment Scenario {run + 1}...")
        print(f"🚥 Active Traffic (Danger Zones): {world.danger_zones}")

        for step in range(20):
            neighbors = world.get_valid_moves(curr)
            
            q_probs = explorer.find_promising_paths(curr, neighbors, world.end_node, world)
            
            # Neural Evaluation
            scores = {}
            for n in neighbors:
                is_danger = 1.0 if n in world.danger_zones else 0.0
                feat = [q_probs.get(n, 0), 0.5, is_danger]
              
                score, _, _ = guide.decide_score(feat)
                scores[n] = score

     
            next_node = max(scores, key=scores.get)
            
            curr = next_node
            path.append(curr)
            
            # Calculate display metrics
            if world.is_end(curr):
                reward = 20.0
            elif curr in world.danger_zones:
                reward = -20.0
            else:
                reward = -0.5
            total_reward += reward
            
            if world.is_end(curr):
                print(f" Goal reached in {step + 1} steps! Total Score: {total_reward:.1f}")
                break
        
        # Update visualization
        viz.update(path, run + 1, total_reward)
        plt.pause(2.5) # Pause to allow path inspection

    print("\n🏁 Deployment Demonstration Complete.")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_test_deployment()
