from typing import Any

from autoevals.string import Levenshtein
from autoevals.number import NumericDiff


def json_subset_match_score(expected: Any, actual: Any) -> float:
    """
    Adapted from autoevals.JsonDiff to only test on the subset of keys within the expected json.
    """
    string_scorer = Levenshtein()
    number_scorer = NumericDiff()
    if isinstance(expected, dict) and isinstance(actual, dict):
        if len(expected) == 0 and len(actual) == 0:
            return 1
        keys = set(expected.keys())
        scores = [json_subset_match_score(expected.get(k), actual.get(k)) for k in keys]
        scores = [s for s in scores if s is not None]
        return sum(scores) / len(scores)
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) == 0 and len(actual) == 0:
            return 1
        scores = [json_subset_match_score(e1, e2) for (e1, e2) in zip(expected, actual)]
        scores = [s for s in scores if s is not None]
        return sum(scores) / max(len(expected), len(actual))
    elif isinstance(expected, str) and isinstance(actual, str):
        return string_scorer.eval(expected, actual).score
    elif (isinstance(expected, int) or isinstance(expected, float)) and (
        isinstance(actual, int) or isinstance(actual, float)
    ):
        return number_scorer.eval(expected, actual).score
    elif expected is None and actual is None:
        return 1
    elif expected is None or actual is None:
        return 0
    else:
        return 0
