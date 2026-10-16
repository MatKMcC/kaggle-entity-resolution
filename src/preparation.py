from functools import cached_property, lru_cache
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
import numpy as np


def _partition_assignment(rng):
    rnd = rng.random()
    if rnd < .1:
        return 'holdout'
    elif rnd < .2:
        return 'validation'
    else:
        return 'train'


class Context:

    # The Constructor Method (initializes each new object)
    def __init__(self, config):
        self.config = config

    # surface the config settings
    # def __getattr__(self, name):
    #   return getattr(self.config, name)
    @lru_cache
    def _load_datasets(self, *datasets):
        # load matchable dataset and filter to unmatched
        tmp = []
        for d in datasets:
            tmp.append(pd.read_csv(d, sep='\t'))
        tmp = pd.concat(tmp, axis=0)
        tmp = tmp.fillna("")
        datasets = {}
        for _id, name, address, country in tmp.values:
            datasets[_id] = {'name': name, 'address': address, 'country': country}
        return datasets

    @cached_property
    def train_entities(self):
        train_datasets = self._load_datasets(self.config['train_entities'])
        print('Loaded.....Train Entities')
        return train_datasets

    @cached_property
    def predict_entities(self):
        train_datasets = self._load_datasets(self.config['predict_entities'])
        print('Loaded.....Predict Entities')
        return train_datasets

    @cached_property
    def train_datasets(self):
        train_datasets = self._load_datasets(*self.config['train_datasets'])
        print('Loaded.....Train Datasets')
        return train_datasets

    @cached_property
    def predict_datasets(self):
        predict_datasets = self._load_datasets(*self.config['predict_datasets'])
        print('Loaded.....Predict Datasets')
        return predict_datasets

    @cached_property
    def known_matches(self):
        tmp = pd.read_csv(self.config['ground_truth'], sep='\t')
        tmp['entity_id'] = tmp['source1_entity_id']
        tmp = tmp[['entity_id', 'matched_entity_ids']]
        tmp.fillna('', inplace=True)
        known_matches = {}
        for entity_id, matches in tmp.values:
            if matches:
                for match in matches.split(','):
                    known_matches[match] = entity_id
        print('Loaded.....Known Matches')
        return known_matches

    @cached_property
    def set_assignments(self):
        if not Path(self.config['set_assignments']).exists():
            # create a list of all observed entities -- grouping matched entities and assigning to partition
            rng = np.random.default_rng()
            tmp = defaultdict(lambda: {'matches': [], 'partition': _partition_assignment(rng)})
            seen_entities = set()
            for matched_entity_id, entity_id in self.known_matches.items():
                tmp[entity_id]['matches'].append(matched_entity_id)
                seen_entities.add(entity_id)
                seen_entities.add(matched_entity_id)
            for entity_id in self.train_entities.keys():
                if entity_id not in seen_entities:
                    tmp[entity_id]['matches'].append(None)
            for unmatched_entity in self.train_datasets.keys():
                if unmatched_entity not in seen_entities:
                    tmp[unmatched_entity]['matches'].append(None)

            # flatten assignments to entity ids
            partitions = {}
            for entity_id in tmp:
                partitions[entity_id] = tmp[entity_id]['partition']
                for matched_entity_id in tmp[entity_id]['matches']:
                    partitions[matched_entity_id] = tmp[entity_id]['partition']

            tmp = pd.DataFrame(partitions.items(), columns=['entity_id', 'partition'])
            tmp.to_csv(self.config['set_assignments'], sep='\t', index=False)
            print('Created.....Set Assignments')
        else:
            tmp = pd.read_csv(self.config['set_assignments'], sep='\t')
            partitions = {e: p for e, p in tmp.values}
            print('Loaded.....Set Assignments')
        return partitions


def summarize_output(context, assignments = None):
    counts = Counter(context.set_assignments.values())
    total_entities = len(context.set_assignments)
    print()
    print('---------------------------------')
    print(f"{'Total Entities Assigned:':<40} {total_entities:>15,.0f}")
    print(f"{'Training Proportion:':<40} {counts['train'] / total_entities:>15,.2f}")
    print(f"{'Validation Proportion:':<40} {counts['validation'] / total_entities:>15,.2f}")
    print(f"{'Holdout Proportion:':<40} {counts['holdout'] / total_entities:>15,.2f}")


def prepare_context(config):
    context = Context(config)
    summarize_output(context)
    return context









