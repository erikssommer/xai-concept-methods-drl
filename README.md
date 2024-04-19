# From Static to Dynamic Concepts in Sequential Decision Making: Using Explainable AI to Improve Reward Functions

Master thesis project in computer science specializing in artificial intelligence at the Norwegian University of Science and Technology (NTNU).

## Description
This project is about creating a deep reinforcement learning agent that can learn to play the game of Go on various board sizes. The agent is trained using the AlphaZero algorithm, and the training is done using the distributed computing framework MPI for high performance computing (HPC). The project also includes a concept detection algorithm that can be used to detect human static and dynamic concepts in the trained agent's policy. The concept detection algorithm is based on the concept activation vectors (CAVs) method. The project also includes a tournament of progressive policies (TOPP) algorithm that can be used to compare the performance of the trained agents.

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

