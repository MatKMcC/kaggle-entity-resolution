import pandas as pd
from featurize import featurize
import numpy as np

from joblib import dump

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

def get_negative_case(entity_list, match_list, reverse_lookup, rng):
    entity, match = '', ''
    while reverse_lookup.get(match, '') == entity:
        entity = rng.choice(entity_list)
        match = rng.choice(match_list)
    return entity, match

if __name__ == '__main__':

    # load entity assignments
    assignments = pd.read_csv('data/source/train/entity_assignments.tsv', sep='\t')
    assignments = {k: v for k, v in assignments.values}
    print('Loaded.....Assignments')

    # load entities data
    tmp = pd.read_csv('data/source/train/source1.tsv', sep='\t')
    tmp = tmp.fillna("")
    entities = {}
    for _id, name, address, country in tmp.values:
        if assignments[_id] == 'train':
            entities[_id] = {'name': name, 'address': address, 'country': country}
    print('Loaded.....Entities')

    # load matchable dataset
    s2 = pd.read_csv('data/source/train/source2.tsv', sep='\t')
    s3 = pd.read_csv('data/source/train/source3.tsv', sep='\t')
    s2['source'] = 'source1'
    s3['source'] = 'source2'
    tmp = pd.concat((s2, s3), axis=0)
    tmp = tmp.fillna("")
    del s2, s3
    datasets = {}
    unknown_assignments = 0
    for _id, name, address, country, source in tmp.values:
        assignment = assignments.get(_id, 'unknown')
        if assignment == 'train':
            datasets[_id] = {'name': name, 'address': address, 'country': country, 'source': source}
        elif assignment == 'unknown':
            unknown_assignments += 1
    print('Loaded.....Match Datasets')
    if unknown_assignments > 0:
        print(f'WARNING: {unknown_assignments:,.0f} unknown assignments found')

    # load reverse lookup
    reverse_lookup = pd.read_csv('data/source/train/reverse_lookup.tsv', sep='\t')
    reverse_lookup = {k:v for k, v in reverse_lookup.values}
    print('Loaded.....Reverse Lookup')

    # generate positive cases
    positive_cases = [(reverse_lookup[k], k) for k in datasets if reverse_lookup.get(k, False)]
    rng = np.random.default_rng()
    rng.shuffle(positive_cases)
    X_pos = pd.DataFrame({'inputs': positive_cases[:100000]})
    X_pos['target'] = 1
    print('Loaded.....Positive Examples')

    # generate the negative cases
    entities_list = np.array(list(entities.keys()))
    match_list = np.array(list(datasets.keys()))
    X_neg = pd.DataFrame({'inputs': range(100000)})
    rng = np.random.default_rng()
    gn = lambda: get_negative_case(entities_list, match_list, reverse_lookup, rng)
    X_neg['inputs'] = [gn() for _ in range(X_neg.shape[0])]
    X_neg['target'] = 0
    print('Loaded.....Negative Examples')

    # combine and reshuffle the dataset
    X = pd.concat((X_pos, X_neg), axis=0)
    X = X.sample(frac=1).reset_index(drop=True)
    print('Loaded.....X Dataset')

    # featurize the dataset
    ftz = lambda x: featurize(entities[x[0]], datasets[x[1]])
    X['features'] = X['inputs'].apply(ftz)
    print('Loaded.....X Features')

    model = LogisticRegression()
    model.fit(X['features'].tolist(), X['target'])
    dump(model, 'data/models/logistic_model.joblib')
    print('Loaded.....Trained Model')

    p = model.predict(X['features'].tolist())
    y = X['target'].tolist()

    print()
    print('---------------------------------')
    print(f"{'Training Precision:':<40} {precision_score(y, p):.2%}")
    print(f"{'Training Recall:':<40} {recall_score(y, p):.2%}")
    print(f"{'Training F1-Score:':<40} {f1_score(y, p):.2%}")

