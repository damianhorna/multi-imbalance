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
# Number of digits used for precision when comparing floats
PRECISION = 0.00001
TP = "tp"
TN = "tn"
FP = "fp"
FN = "fn"
OPPOSITE_LABEL_TO_RULE = "opposite"
ALL_LABELS = "all"
SAME_LABEL_AS_RULE = "same"
# {example ei: rule ri for which ei is the seed}
seed_example_rule = {}
# {rule ri: example ei is seed for ri}
seed_rule_example = {}
# {rule ri: set(example ei)}
examples_covered_by_rule = {}
# {example ei: tuple(rule ri, distance di)}
closest_rule_per_example = {}
# {rule ri: set(example ei, example ej)}
closest_examples_per_rule = {}
conf_matrix = {TP: set(), FP: set(), TN: set(), FN: set()}
# {hash of rule ri: ID of rule ri}
unique_rules = {}
# {ID of rule ri: rule ri (=pd.Series)}
all_rules = {}
latest_id = 0
minority_class = ""
