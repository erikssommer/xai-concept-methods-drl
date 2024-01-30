# go-xai
Master Specialization Project

### Install required packages
```bash
pip install -r requirements.txt
```

### Set the environment variables in the config file
```bash
nano config.py
```

### Train models single threaded
> From the src directory
```bash
python train_single_thread.py
```

### Train models multi threaded (MPI)
> From the src directory
```bash
mpirun -np 4 python train_hpc.py
```

### Play against the trained models
> From the src directory
```bash
python play.py
```

### Topp - tournament of progressive policies
> From the src directory
```bash
python tournament.py
```

### Run the tests
> From the tests directory
```bash
python test_name.py
```

### Run training on Idun
> From the root directory
```bash
sbatch hpc.sh
```

### Tensorboard
> From the root directory
```bash
tensorboard --logdir tensorboard_logs/
```

### Concept Activation Vectors
> From the notebooks directory

Run concept_detection.ipynb to get the concept activation vectors

Run concept_detection.ipynb to get the concept activation vectors visualized

Run model_performance.ipynb to get the model performance

