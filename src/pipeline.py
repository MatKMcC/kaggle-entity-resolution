import yaml

from preparation import prepare_context
from train import train_model
from predict import make_matches
from evaluate import evaluate_matches

def config(config_path):
    yaml_file = yaml.safe_load(open(config_path, 'r'))
    return yaml_file

def pipeline(config):
    print()
    print('pipeline started')
    context = prepare_context(config)
    train_model(context)
    make_matches(context)
    evaluate_matches(context)
    print('pipeline finished')