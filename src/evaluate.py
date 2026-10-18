import pickle
from pathlib import Path
import argparse

from preparation import Context

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
    print(f'{'-' * 71}')
    print(f"{'Matches Found:':<40} {_metrics['matches']:>30,.0f}")
    print(f"{'Precision Score:':<40} {_metrics['precision']:>30,.2%}")
    print(f"{'Recall Score:':<40} {_metrics['recall']:>30,.2%}")
    print(f"{'F1 Score:':<40} {_metrics['f1_score']:>30,.2%}")
    print(f'{'-' * 71}')
    print()


def evaluate(ground_truth, predictions, combinations_checked, run_time):

    metrics = {
        'precision': precision_score(ground_truth, predictions)
        , 'recall': recall_score(ground_truth, predictions)
        , 'f1_score': f1_score(ground_truth, predictions)
        , 'matches': len(predictions)
    }

    return metrics

def evaluate_matches(context):

    # load match data
    pth = Path(context.config['predicted_matches'])
    with open(str(pth / 'predictions.pkl'), 'rb') as f:
        tmp = pickle.load(f)
    combinations_checked = tmp.pop('combinations_checked')
    run_time = tmp.pop('runtime')
    total_combinations = tmp.pop('total_combinations')
    search_combinations = tmp.pop('search_combinations')
    predictions = set()
    for entity in tmp:
        for match in tmp[entity]:
            predictions.add((entity, match))

    print('Total Prediction Run Time Metrics')
    print(f'{'-' * 71}')
    print(f'{'Run Time Metrics For Prediction Set:'}')
    print(f"{'Combinations Checked:':<40} {combinations_checked:>30,.2f}")
    print(f"{'Total Possible Combinations:':<40} {total_combinations:>30,.2f}")
    print(f"{'Search Efficiency:':<40} {len(predictions) / combinations_checked:>30,.2%}")
    print(f"{'Reduction Score:':<40} {(search_combinations - combinations_checked) / search_combinations:>30,.2%}")
    print(f"{'Run Time:':<40} {run_time / 60.:>26,.2f} min")
    print(f'{'-' * 71}')
    print()

    # iterate over partitions and generate metrics
    for partition in context.config['evaluate_datasets']:
        print(f"Evaluation Metrics for {partition}")

        set_known_matches = {(k, v) for k, v in context.known_matches.items() if context.set_assignments[k] == partition}
        set_predictions = {el for el in predictions if context.set_assignments[el[0]] == partition}
        # evaluate
        metrics = evaluate(set_known_matches, set_predictions, combinations_checked, run_time)
        display_metrics(metrics, partition)

        # save results
        pth = Path(context.config['evaluation_metrics'])
        pth.mkdir(parents=True, exist_ok=True)
        with open(str(pth / 'metrics.pkl'), 'wb') as f:
            pickle.dump(metrics, f)

    return None


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', required=True)
    args = arg_parser.parse_args()

    context = Context(args.config)
    evaluate_matches(context)
