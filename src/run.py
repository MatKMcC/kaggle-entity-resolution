from pipeline import pipeline
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config'
        , default='./config.yaml')
    args = parser.parse_args()
    metrics = pipeline(args.config)