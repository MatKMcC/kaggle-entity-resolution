import Levenshtein
import numpy as np


def _basic_string_cleaning(string):
    return string.strip().lower()


def _clean_business_name(name):
    return _basic_string_cleaning(name)


def _clean_address(address):
    return _basic_string_cleaning(address)


def _characters(string):
    return len(string)


def _words(string):
    return len(set(string.split(' ')))


def _same_check(v1, v2):
    return float(v1 == v2)


def _distance(v1, v2):
    return Levenshtein.distance(v1, v2)


def _word_match(v1, v2):
    v1 = set(v1.split(' '))
    v2 = set(v2.split(' '))
    return len(v1.intersection(v2))


def featurize(entity, match):

    features = []

    entity_name = _basic_string_cleaning(entity['name'])
    entity_address = _basic_string_cleaning(entity['address'])
    match_name = _basic_string_cleaning(match['name'])
    match_address = _basic_string_cleaning(match['address'])

    # check if string lengths are close. Note this is more valuable as an interactive feature
    max_name_string_length = max(_characters(entity_name), _characters(match_name), 1)
    max_address_string_length = max(_characters(entity_address), _characters(match_address), 1)

    # check if word lengths are close. Note this is more valuable as an interactive feature
    max_name_word_length = max(_words(entity_name), _words(match_name), 1)
    max_address_word_length = max(_words(entity_address), _words(match_address), 1)


    # run comparison features ---------
    # exact check on fields
    features.append(_same_check(entity_name, match_name))
    features.append(_same_check(entity_address, match_address))

    # check txt distance
    features.append(_distance(entity_name, match_name) / max_name_string_length)
    features.append(_distance(entity_address, match_address) / max_address_string_length)

    # check word matches
    features.append(_word_match(entity_name, match_name) / max_name_word_length)
    features.append(_word_match(entity_address, match_address) / max_address_word_length)

    return np.array(features)
