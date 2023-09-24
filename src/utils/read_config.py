import yaml
import argparse

# Importing config settings

with open('../config.yml', 'r') as file:
    config_data = yaml.safe_load(file)

config = argparse.Namespace(**config_data)