import numpy as np

MISSING_VAL = np.nan
NOMINAL = 0
NUMERIC = 1
CONDITIONAL = "Conditional"
COVERED = "is_covered"
CLASSES = []
SAFE = "safe"
NOISY = "noisy"
BORDERLINE = "borderline"
TAG = "tag"
DIST = "dist"
DISTANCE_MATRIX = {}
TP = "tp"
TN = "tn"
FP = "fp"
FN = "fn"
OPPOSITE_LABEL_TO_RULE = "opposite"
ALL_LABELS = "all"
SAME_LABEL_AS_RULE = "same"
# {example ei: seed for rule ri}
seed_mapping = {}
# {rule ri: set(example ei)}
examples_covered_by_rule = {}
# {rule ri: set(closest rule for examples ei, ej)]}
predicted_examples = {}
# {example ei: (rule ri, distance di )}
closest_rule_per_example = {}
conf_matrix = {TP: set(), FP: set(), TN: set(), FN: set()}
latest_id = 0
positive_class = ""
