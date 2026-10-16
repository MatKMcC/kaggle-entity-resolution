import argparse
import pandas as pd
import pickle
from pathlib import Path
from collections import defaultdict


def _true_positives(ground_truth, predictions):
    return len(ground_truth.intersection(predictions))


def _false_positives(ground_truth, predictions):
    return len(predictions - ground_truth)


def _false_negatives(ground_truth, predictions):
    return len(ground_truth - predictions)


def precision_score(ground_truth, predictions):
    tp = _true_positives(ground_truth, predictions)
    fp = _false_positives(ground_truth, predictions)
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)


def recall_score(ground_truth, predictions):
    tp = _true_positives(ground_truth, predictions)
    fn = _false_negatives(ground_truth, predictions)
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)


def f1_score(ground_truth, predictions):
    p = precision_score(ground_truth, predictions)
    r = recall_score(ground_truth, predictions)
    if p + r == 0:
        return 0
    return 2 * p * r / (p + r)


def display_metrics(_metrics, _partition):

    print(f'{'Evaluating Results for Dataset:':<40}  {_partition:>15}')
    print(f"{'Combinations Checked:':<40} {_metrics['combinations']:>15,.2f}")
    print(f"{'Matches Found:':<40} {_metrics['matches']:>15,.0f}")
    print(f"{'Efficiency:':<40} {_metrics['efficiency']:>15,.2%}")
    print(f"{'Run Time:':<40} {_metrics['run_time']:>11,.2f} min")
    print(f"{'Precision Score:':<40} {_metrics['precision']:>15,.2%}")
    print(f"{'Recall Score:':<40} {_metrics['recall']:>15,.2%}")
    print(f"{'F1 Score:':<40} {_metrics['f1_score']:>15,.2%}")


def evaluate(ground_truth, predictions, combinations_checked, run_time):

    metrics = {
        'precision': precision_score(ground_truth, predictions)
        , 'recall': recall_score(ground_truth, predictions)
        , 'f1_score': f1_score(ground_truth, predictions)
        , 'combinations': combinations_checked
        , 'matches': len(predictions)
        , 'efficiency': len(predictions) / combinations_checked
        , 'run_time': run_time / 60.
    }

    return metrics


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-partitions",
        choices=["train", "validation", "holdout", "unassigned"],
        default=["train", "validation", "holdout"],
        help="training set choices")
    args = parser.parse_args()

    # load assignments
    pth = Path(f'data/source/train/entity_assignments.tsv')
    if pth.exists():
        tmp = pd.read_csv(pth, sep='\t')
        assignments = {entity: match for entity, match in tmp.values}
    else:
        assignments = {}
    print('Loaded.....Assignments')

    # load ground truth data
    tmp = pd.read_csv('data/source/train/ground_truth.tsv', sep='\t')
    tmp = tmp.fillna('')
    ground_truth = defaultdict(set)
    for (entity, matches) in tmp.values:
        for match in matches.split(','):
            if match:
                partition = assignments.get(entity, 'unassigned')
                ground_truth[partition].add((entity, match))
    print('Loaded.....Ground Truth')

    # iterate over partitions and generate metrics
    for partition in args.data_partitions:
        print('---------------------------------')

        # load match data
        pth = Path(f'data/predictions/train/{partition}/matches.pkl')
        if not pth.exists():
            print(f"The data partition does not exist: '{partition}'")
        else:
            with open(str(pth), 'rb') as f:
                tmp = pickle.load(f)
            combinations_checked = tmp.pop('combinations_checked')
            run_time = tmp.pop('runtime')
            predictions = set()
            for entity in tmp:
                for match in tmp[entity]:
                    predictions.add((entity, match))

            # evaluate
            metrics = evaluate(ground_truth[partition], predictions, combinations_checked, run_time)
            display_metrics(metrics, partition)

            # save results
            with open(f'data/predictions/train/{partition}/metrics.pkl', 'wb') as f:
                pickle.dump(metrics, f)

