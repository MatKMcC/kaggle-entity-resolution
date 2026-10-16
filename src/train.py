import pandas as pd
from .featurize import featurize
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

def train_model(context):
    entities = {_id: values for _id, values in context.train_entities.items() if context.set_assignments[_id] == 'train'}
    datasets = {_id: values for _id, values in context.train_datasets.items() if context.set_assignments[_id] == 'train'}

    # generate positive cases
    positive_cases = [(context.known_matches[k], k) for k in datasets if context.known_matches.get(k, False)]
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
    gn = lambda: get_negative_case(entities_list, match_list, context.known_matches, rng)
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
    dump(model, context.config['model_path'])
    print('Loaded.....Trained Model')

    p = model.predict(X['features'].tolist())
    y = X['target'].tolist()

    print()
    print('---------------------------------')
    print(f"{'Training Precision:':<40} {precision_score(y, p):.2%}")
    print(f"{'Training Recall:':<40} {recall_score(y, p):.2%}")
    print(f"{'Training F1-Score:':<40} {f1_score(y, p):.2%}")

    return None
