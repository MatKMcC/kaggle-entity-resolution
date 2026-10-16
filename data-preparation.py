import pandas as pd
import numpy as np
from collections import Counter


def _partition_assignment(rnd):
    if rnd < .1:
        return 'holdout'
    elif rnd < .2:
        return 'validation'
    else:
        return 'train'


if __name__ == "__main__":

    # load ground truth and remove null values
    ground_truth = pd.read_csv('data/source/train/ground_truth.tsv', sep ='\t')
    ground_truth['entity_id'] = ground_truth['source1_entity_id']
    ground_truth = ground_truth[['entity_id', 'matched_entity_ids']]
    ground_truth.fillna('', inplace=True)
    print('Loaded.....Ground Truth')

    # create reverse lookup table
    reverse_lookup = []
    matched_entities = set()
    for entity, matched in ground_truth.values:
        matched_entities.add(entity)
        if matched != '':
            for match in matched.split(','):
                reverse_lookup.append({'matched_entity': match, 'entity_id': entity})
                matched_entities.add(match)
    tmp = pd.DataFrame(reverse_lookup)
    tmp.to_csv('./data/source/train/reverse_lookup.tsv', sep = '\t', index = False)
    print('Loaded.....Reverse Lookup')

    # load entities dataset and filter to unmatched
    entities = pd.read_csv('data/source/train/source1.tsv', sep='\t')
    entities = entities.fillna("")
    unmatched_entities = entities[entities['entity_id'].apply(lambda el: el not in matched_entities)]
    unmatched_entities['matched_entity_ids'] = ''
    unmatched_entities = unmatched_entities[['entity_id', 'matched_entity_ids']]
    del entities
    print('Loaded.....Unmatched in Entities')

    # load matchable dataset and filter to unmatched
    s2 = pd.read_csv('data/source/train/source2.tsv', sep='\t')
    s3 = pd.read_csv('data/source/train/source3.tsv', sep='\t')
    s2['source'] = 'source1'
    s3['source'] = 'source2'
    datasets = pd.concat((s2, s3), axis=0)
    datasets = datasets.fillna("")
    unmatched_datasets = datasets[datasets['entity_id'].apply(lambda el: el not in matched_entities)]
    unmatched_datasets['entity_id'] = unmatched_datasets['entity_id']
    unmatched_datasets['matched_entity_ids'] = ''
    unmatched_datasets = unmatched_datasets[['entity_id', 'matched_entity_ids']]
    del s2, s3, datasets
    print('Loaded.....Unmatched in Datasets')

    # combine to a single dataset
    entities = pd.concat((ground_truth, unmatched_entities, unmatched_datasets), axis=0)

    # randomly assign entity_ids (entities and their match are in the same partition)
    rng = np.random.default_rng()
    entities['assignment'] = rng.random(entities.shape[0])
    entities['assignment'] = entities['assignment'].apply(_partition_assignment)
    print('Loaded.....Combined Datasets')

    assignments = []
    counts = Counter()
    for entity_id, matched_entities, assignment in entities.values:
        assignments.append({'entity_id': entity_id, 'assignment': assignment})
        counts[assignment] += 1
        if matched_entities != '':
            for matched_entity in matched_entities.split(','):
                assignments.append({'entity_id': matched_entity, 'assignment': assignment})
                counts[assignment] += 1
    tmp = pd.DataFrame(assignments)
    tmp.to_csv('./data/source/train/entity_assignments.tsv', sep = '\t', index = False)
    print('Loaded.....Assignments')

    print()
    print('---------------------------------')
    print(f"{'Total Entities Assigned:':<40} {len(assignments):>15,.0f}")
    print(f"{'Training Proportion:':<40} {float(counts['train']) / float(len(assignments)):>15,.2f}")
    print(f"{'Validation Proportion:':<40} {float(counts['validation']) / float(len(assignments)):>15,.2f}")
    print(f"{'Holdout Proportion:':<40} {float(counts['holdout']) / float(len(assignments)):>15,.2f}")





