from topology import NetworkTopology
from quantum_engine import QuantumExplorer
from brain import NeuralGuide
from visualizer import CityVisualizer
from intelligence_manager import IntelligenceManager
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

def train():
    world = NetworkTopology()
    explorer = QuantumExplorer()
    # Using 12 hidden neurons gives the brain slightly more memory for traffic patterns
    guide = NeuralGuide(input_size=3, hidden_size=12) 
    
    # Load previous knowledge at the start
    IntelligenceManager.load_intelligence(guide)
    
    viz = CityVisualizer(world)
    memory = deque(maxlen=2000)
    
 
    episodes, epsilon, gamma = 200, 1.0, 0.95
    decay = 0.99 

    print("🚀 Initializing Dynamic Quantum-Neural Routing...")

    for ep in range(episodes):
        world.update_traffic() 
        curr = world.start_pos
        path, total_reward = [curr], 0
        
        for step in range(25):
            neighbors = world.get_valid_moves(curr)
            # Reducing shots in quantum_engine.py to 25 is recommended to avoid hangs
            q_probs = explorer.find_promising_paths(curr, neighbors, world.end_node, world)
            
            scores = {}
            for n in neighbors:
                is_danger = 1.0 if n in world.danger_zones else 0.0
                # Feature vector: [Quantum Suggestion, Default Bias, Danger Sensor]
                feat = [q_probs.get(n, 0), 0.5, is_danger]
                score, _, _ = guide.decide_score(feat)
                scores[n] = (score, feat)

            if np.random.rand() < epsilon:
                next_node = np.random.choice(neighbors)
            else:
                next_node = max(scores, key=lambda k: scores[k][0])
            
         
            if world.is_end(next_node):
                reward = 20.0
            elif next_node in world.danger_zones:
                reward = -20.0
            else:
                reward = -0.5 
                
            total_reward += reward
            memory.append((scores[next_node][1], reward, next_node, world.is_end(next_node)))
            
            curr = next_node
            path.append(curr)
            if world.is_end(curr): break

     
        viz.update(path, ep + 1, total_reward)

   
        if len(memory) > 64:
            batch = [memory[i] for i in np.random.choice(len(memory), 64)]
            for feat, r, nxt, done in batch:
                pred, h, ins = guide.decide_score(feat)
                target = r if done else r + gamma * guide.decide_score(feat)[0]
                guide.learn(target, pred, h, ins)

        epsilon = max(0.01, epsilon * decay)
        if (ep + 1) % 10 == 0:
            print(f"Ep {ep+1} | Reward: {total_reward:.1f} | Epsilon: {epsilon:.2f}")

    # Final Save
    IntelligenceManager.save_intelligence(guide)
    print(" Training Complete. Intelligence stored.")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    train()
