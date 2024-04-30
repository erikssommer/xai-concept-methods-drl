# From Static to Dynamic Concepts in Sequential Decision Making: Improving Reward Functions Using Explainable AI

Codebase for master thesis project in computer science specializing in artificial intelligence at the Norwegian University of Science and Technology (NTNU).

## Description
This project aims to improve the reward functions of deep reinforcement learning agents by using explainable AI to detect static and dynamic concepts in the agent's policy.

The codebase contains:
*   the deep reinforcmemt training loop, similar to the one outlined in the AlphaZero paper by Google DeepMind
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

### Run training on HPC (Idun is used in this project)
```bash
sbatch hpc.sh
```

### Tensorboard - visualize the training
```bash
tensorboard --logdir tensorboard_logs/
```

### Concept Activation Vectors
Run ```static_concept_detection.ipynb``` to detect static concepts in the trained agent's policy.

Run ```dynamic_concept_detection.ipynb``` to detect dynamic concepts in the trained agent's policy.

Run ```concept_detection.ipynb``` to get the concept activation vectors visualized

### Model performance
Run ```model_performance.ipynb``` to pit the trained agents against each other and compare their performance.

