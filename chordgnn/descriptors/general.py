import numpy as np
import partitura.score
from .utils import *
from typing import Union, Tuple, List


def chord_analysis_features(part: Union[Union[partitura.score.Part, partitura.score.PartGroup], partitura.performance.PerformedPart]) -> Tuple[np.ndarray, List]:
    features, fnames = get_chord_analysis_features(part)
    return features, fnames


def select_features(part, features):
    note_features, _ = chord_analysis_features(part)
    return note_features
