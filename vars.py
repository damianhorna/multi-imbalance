import numpy as np

MISSING_VAL = np.nan
NOMINAL = 0
NUMERIC = 1
CONDITIONAL = "Conditional"
CLASSES = []
SAFE = "safe"
NOISY = "noisy"
BORDERLINE = "borderline"
TAG = "tag"
DISTANCE_MATRIX = {}
TP = "tp"
TN = "tn"
FP = "fp"
FN = "fn"
# {example ei: seed for rule ri}
seed_mapping = {}
# {rule ri: set(closest rule for examples ei, ej)]}
predicted_examples = {}
# {example ei: predicted by rule ri}
closest_rule = {}
conf_matrix = {TP: set(), FP: set(), TN: set(), FN: {}}
latest_id = 0