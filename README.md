#  Quantum-Neural City Agent: Dynamic Path Optimization

A hybrid AI system that combines **Quantum Computing (Grover's Algorithm)** with **Deep Reinforcement Learning** to navigate a non-stationary city environment. This project demonstrates how quantum "intuition" can be combined with neural "experience" to solve complex, real-time routing problems.

##  Key Features

* **Hybrid Intelligence**: Integrates **Qiskit** for quantum path suggestions and a custom **Neural Network** for safety evaluation.
* **Non-Stationary Environment**: Traffic "Danger Zones" move every episode, forcing the agent to adapt its logic in real-time.
* **Performance Dashboard**: A dual-pane real-time visualizer showing the agent's pathfinding and its learning curve simultaneously.
* **Persistence Engine**: Automated saving and loading of neural weight matrices (, ) for persistent intelligence across sessions.

---

##  System Architecture

The agent operates across three distinct layers:

### 1. Symbolic Layer (`topology.py`)

Uses Graph Theory to define the city's structure. It manages nodes (intersections) and edges (roads), providing the "rules" of the world to the agent.

### 2. Quantum Layer (`quantum_engine.py`)

Implements a **Grover-inspired Amplitude Amplification** algorithm. It uses a quantum oracle to "mark" mathematically shortest paths in a superposition of all possible moves, providing the agent with a "quantum intuition" for efficiency.

### 3. Neural Layer (`brain.py`)

A custom-built neural network that acts as the agent's experience-based brain. It takes quantum suggestions and danger-sensor data to calculate a "Safe-Score" for every move, learning from penalties when it hits traffic.



##  Performance Results

During training, the agent demonstrates a classic reinforcement learning "S-Curve":

* **Exploration (Ep 1-100)**: The agent moves randomly (high Epsilon), learning the high cost of hitting Red Danger Zones (-20 penalty).
* **Breakthrough (Ep 110-150)**: The neural network begins to consistently override the "Shortest Path" suggestion if it leads to danger, resulting in a positive reward spike (reaching **+18.5**).
* **Deployment (Test Mode)**: With randomness (Epsilon) set to 0, the agent utilizes 100% of its learned intelligence to navigate 100% of scenarios successfully.

<img width="1853" height="813" alt="Screenshot 2026-01-21 010434" src="https://github.com/user-attachments/assets/8c1168d9-de21-43b7-8828-18bce34d2467" />

<img width="1865" height="774" alt="Screenshot 2026-01-21 010250" src="https://github.com/user-attachments/assets/151e8ef5-4788-46cd-8463-df3d7fd02238" />


##  Installation & Usage

### Prerequisites

* Python 3.10+
* Virtual Environment recommended

### Setup

```powershell
# Install dependencies
pip install numpy networkx qiskit qiskit_aer matplotlib

```

### Run Training

```powershell
python main.py

```

### Run Deployment (Test)

```powershell
python test_agent.py

```

---

## Project Structure

* `main.py`: The training loop and experience replay manager.
* `test_agent.py`: Exploitation script for showing optimized performance.
* `quantum_engine.py`: Qiskit implementation of quantum pathfinding logic.
* `brain.py`: Neural Network architecture and learning algorithms.
* `visualizer.py`: Matplotlib-based performance dashboard.
* `intelligence_manager.py`: Logic for saving/loading `.npz` weight files.


