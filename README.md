# go-xai
Master Specialization Project

### Install required packages
```bash
pip install -r requirements.txt
```

### Run the application
```bash
cd src
python main.py
```

### Run the tests
> From the root directory
```bash
python -m unittest tests."filename"
```
> Example
```bash
python -m unittest tests.random_play_test
```

### Run the training on the server in the background with logs
> From src directory
```bash
nohup python train.py > train_log.txt &
```

