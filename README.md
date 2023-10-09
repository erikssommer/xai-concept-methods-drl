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

### Train models
```bash
cd src
python train.py
```

### Play against the trained models
```bash
cd src
python play.py
```

### Topp - tournament of progressive policies
```bash
cd src
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
sbatch job.slurm
```

### Run the training on the server in the background with logs
> From src directory
```bash
nohup python train.py > train_log.txt &
nohup python train_multi_threading.py > train_log.txt &
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

