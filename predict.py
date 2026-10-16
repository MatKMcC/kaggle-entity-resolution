'''
This is the one! Not sure how to do this efficiently, we need to cover all combinations
or find efficient ways to handle them
'''
from featurize import featurize
from collections import defaultdict
import pandas as pd
from joblib import load
import time
import pickle
from pathlib import Path
import numpy as np
import argparse


def run_predict(_source, _dataset, _entities, _matches, _runtime_cutoff):
    # start timer -- report time and break when timer hits 10 minutes
    start_time = time.perf_counter()
    minutes = 0
    counts = 0
    matches_found = defaultdict(list)

    for match in _matches:
        for entity in _entities:

            counts += 1
            features = featurize(_entities[entity], _matches[match])
            features = features.reshape(1, -1)
            p = quick_predict(features)

            if round((time.perf_counter() - start_time) / 60.) != minutes:
                print(f"Executed in: {(time.perf_counter() - start_time):,.0f} seconds")
                print(f"Combinations Checked: {counts:,.0f}")
                print(f"Matches Found: {sum([len(matches_found[el]) for el in matches_found]):,.0f}")
                minutes = round((time.perf_counter() - start_time) / 60.)

            # match found! add and skip future entity checks for this match
            if p:
                matches_found[entity].append(match)
                break

            if (time.perf_counter() - start_time) / 60. > _runtime_cutoff:
                print(f"{int(_runtime_cutoff)} Minute Timeout")
                return results(_source, _dataset, matches_found, counts, start_time)

    return results(_source, _dataset, matches_found, counts, start_time)


def results(_source, _dataset, _matches_found, counts, start_time):
    print()
    print('------------------------------------------------')
    print(f"Combinations Checked: {counts:,.0f}")
    print(f"Matches Found: {sum([len(_matches_found[el]) for el in _matches_found]):,.0f}")
    _matches_found['combinations_checked'] = counts
    _matches_found['runtime'] = time.perf_counter() - start_time
    pth = Path(f"data/predictions/{_source}/{_dataset}/")
    pth.mkdir(parents=True, exist_ok=True)
    with open(str(pth / 'matches.pkl'), "wb") as f:
        pickle.dump(_matches_found, f)


if __name__ == '__main__':

    # assign variables
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["train", "test"],
        required=True,
        default="train",
        help="choice for the data source")
    parser.add_argument(
        "--data-partitions",
        choices=["train", "validation", "holdout", "unassigned"],
        default=["train", "validation", "holdout"],
        help="training set choices")
    parser.add_argument(
        "--runtime-cutoff",
        default=10,
        help="training set choices")
    args = parser.parse_args()
    source = args.source
    data_partitions = args.data_partitions
    runtime_cutoff = float(args.runtime_cutoff)
    if source == "test":
        data_partitions = ["unassigned"]

    # load entity dataset
    tmp = pd.read_csv(f'data/source/{source}/source1.tsv', sep='\t')
    tmp = tmp.fillna("")
    entities = {}
    for _id, name, address, country in tmp.values:
        entities[_id] = {'name': name, 'address': address, 'country': country}
    print('Loaded.....Entities')

    # load match datasets
    s2 = pd.read_csv(f'data/source/{source}/source2.tsv', sep='\t')
    s2['source'] = 'source1'
    s3 = pd.read_csv(f'data/source/{source}/source3.tsv', sep='\t')
    s3['source'] = 'source2'
    tmp = pd.concat((s2, s3), axis=0)
    tmp = tmp.fillna("")
    del s2, s3
    matches = {}
    for _id, name, address, country, _source in tmp.values:
        matches[_id] = {'name': name, 'address': address, 'country': country, 'source': _source}
    print('Loaded.....Match Datasets')

    # load assignments
    pth = Path(f'data/source/{source}/entity_assignments.tsv')
    if pth.exists():
        tmp = pd.read_csv(pth, sep='\t')
        assignments = {entity: match for entity, match in tmp.values}
    else:
        assignments = {}

    # partition datasets in assignments
    datasets = defaultdict(lambda: defaultdict(dict))
    for entity in entities:
        assignment = assignments.get(entity, 'unassigned')
        if assignment in data_partitions:
            datasets[assignments.get(entity, 'unassigned')]['entities'][entity] = entities[entity]
    for entity in matches:
        datasets[assignments.get(entity, 'unassigned')]['matches'][entity] = matches[entity]
    del entities
    del matches

    # load model and build quick prediction
    model = load('data/models/logistic_model.joblib')
    coef = model.coef_
    intercept = model.intercept_
    def quick_predict(x):
        # Dot product + Sigmoid
        z = np.dot(x, coef.T) + intercept
        p = 1 / (1 + np.exp(-z))
        return 1.0 if p > 0.5 else 0.0
    print('Loaded.....Model')

    # iterate over combinations
    for dataset in datasets:
        entities = datasets[dataset]['entities']
        matches = datasets[dataset]['matches']
        run_predict(source, dataset, entities, matches, runtime_cutoff)





