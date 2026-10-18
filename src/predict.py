'''
This is the one! Not sure how to do this efficiently, we need to cover all combinations
or find efficient ways to handle them
'''
from collections import defaultdict
import pandas as pd
from joblib import load
import time
import pickle
from pathlib import Path
import numpy as np
import argparse

from preparation import Context
from featurize import featurize

def run_predict(entities, datasets, runtime_cutoff, quick_predict):
    # iterate over combinations to find a match
    counts = 0
    minutes = 0
    start_time = time.perf_counter()
    matches_found = defaultdict(list)

    search_combinations = 0
    for potential_match in datasets:
        search_combinations += len(entities)
        for entity in entities:

            counts += 1
            features = featurize(entities[entity], datasets[potential_match])
            features = features.reshape(1, -1)
            p = quick_predict(features)

            if round((time.perf_counter() - start_time) / 60.) != minutes:
                print(f"Executed in: {(time.perf_counter() - start_time):,.0f} seconds")
                print(f"Combinations Checked: {counts:,.0f}")
                print(f"Matches Found: {len(matches_found):,.0f}")
                minutes = round((time.perf_counter() - start_time) / 60.)

            # match found! add and skip future entity checks for this match
            if p:
                matches_found[entity].append(potential_match)
                break

            if (time.perf_counter() - start_time) / 60. > runtime_cutoff:
                print(f"{int(runtime_cutoff)} Minute Timeout")
                return matches_found, counts, start_time, len(datasets) * len(entities), search_combinations
    return matches_found, counts, start_time, len(datasets) * len(entities), search_combinations


def make_matches(context):

    # load model and build quick prediction
    model = load(context.config['model_path'])
    coef = model.coef_
    intercept = model.intercept_
    def quick_predict(x):
        # Dot product + Sigmoid
        z = np.dot(x, coef.T) + intercept
        p = 1 / (1 + np.exp(-z))
        return 1.0 if p > 0.5 else 0.0
    print('Loaded.....Model')

    # load datasets and params
    datasets = context.predict_datasets
    entities = context.predict_entities
    runtime_cutoff = float(context.config['runtime_cutoff'])
    matches_found, counts, start_time, total_combinations, search_combinations = run_predict(entities, datasets, runtime_cutoff, quick_predict)

    print()
    print('Predicted Matches for Entities')
    print(f'{'-' * 71}')
    print(f"{'Combinations Checked:':<40} {counts:>30,.0f}")
    print(f"{'Matches Found:':<40} {sum([len(matches_found[el]) for el in matches_found]):>30,.0f}")
    print(f'{'-' * 71}')
    print()

    matches_found['combinations_checked'] = counts
    matches_found['runtime'] = time.perf_counter() - start_time
    matches_found['total_combinations'] = total_combinations
    matches_found['search_combinations'] = search_combinations
    pth = Path(context.config['predicted_matches'])
    pth.mkdir(parents=True, exist_ok=True)
    with open(str(pth / 'predictions.pkl'), "wb") as f:
        pickle.dump(matches_found, f)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', required=True)
    args = arg_parser.parse_args()

    context = Context(args.config)
    make_matches(context)





