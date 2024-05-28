# From Static to Dynamic Concepts in Sequential Decision Making: Improving Reward Functions Using Explainable AI

Codebase for master thesis project in computer science specializing in artificial intelligence at the Norwegian University of Science and Technology (NTNU).

## Description
This project aims to discover the aquisition of static and dynamic concepts in the policy of an agent trained using deep reinforcement learning. The project is based on the AlphaGo Zero algorithm by Google DeepMind, and the agent is trained in the game of Go. The project uses Concept Activation Vectors (CAVs) to find static and dynamic concepts in the agent's policy. The project also uses the Monte Carlo Tree Search (MCTS) algorithm to unsupervisedly generate datasets for dynamic concepts. The project uses a joint embedding model to learn the relationship between state-action pairs and conceptual explanations. The project uses the joint embedding model and concept functions to improve the reward function of the agent. The project also trains a concept bottleneck model to learn concepts in the agent's policy.

The codebase contains:
*   the deep reinforcmemt training loop, similar to the one outlined in the AlphaGo Zero paper by Google DeepMind
*   concept detection using Concept Activation Vectors (CAVs) to find static and dynamic concepts in the agent's policy
*   concept functions for static concepts
*   algorithm using the monte carlo tree search (MCTS) to unsupervisedly generate datasets for dynamic concepts
*   joint embedding model to learn the relationship between state-action pairs and conceptual explanations
*   using the joint embedding model and concept functions to improve the reward function of the agent
*   Training a concept bottleneck model to learn concepts in the agent's policy


### Install required packages
```bash
python -m pip install -r requirements.txt
```

### Set the environment variables in the config file
```bash
nano config.py
```

### Train models single threaded
```bash
python train_single_thread.py
```

### Train models multi threaded (MPI)
```bash
mpirun -np 4 python train_hpc.py
```

### Play against the trained models
```bash
python play.py
```

### Topp - tournament of progressive policies
```bash
python tournament.py
```

### Run the tests
```bash
python test_name.py
```

### Run training on HPC (Idun at NTNU is used in this project)
```bash
sbatch hpc.sh
```

### Tensorboard - visualize the training
```bash
tensorboard --logdir tensorboard_logs/
```

### Results from the experiments

The results from the experiments are located in the `notebooks` folder. The notebooks are named according to the experiments they represent.

