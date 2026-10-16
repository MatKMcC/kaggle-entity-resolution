from src import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config'
        , default='./config.yaml')
    args = parser.parse_args()
    print(args.config)
    cfg = config(args.config)
    metrics = pipeline(cfg)