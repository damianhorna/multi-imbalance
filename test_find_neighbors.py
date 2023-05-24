from unittest import TestCase
from collections import Counter

import pandas as pd

import scripts.vars as my_vars
from scripts.utils import find_nearest_examples, Data, Bounds


class TestFindNeighbors(TestCase):
    """Test find_neighbors() from utils.py"""

    def test_find_neighbors_too_few(self):
        """Test that warning is thrown if too few neighbors exist"""
        dataset = pd.DataFrame({"A": [1, 2], "B": [1, 2], "C": [2, 2], "D": ["x", "y"], "Class": ["A", "B"]})
        rule = pd.Series({"A": (0.1, 1), "B": Bounds(lower=1, upper=1), "C": Bounds(lower=2, upper=2), "D": "x",
                          "Class": "A"})
        k = 3
        classes = ["apple", "banana"]
        class_col_name = "Class"
        min_max = pd.DataFrame({"A": {"min": 1, "max": 5}, "B": {"min": 1, "max": 11}, "C": {"min": 1, "max": 2}})
        lookup = \
            {
                "D":
                    {
                        'x': 1,
                        'y': 1,
                        my_vars.CONDITIONAL:
                            {
                                'x':
                                    Counter({
                                        'A': 1
                                    }),
                                'y':
                                    Counter({
                                        'B': 1
                                    })
                            }
                    }
            }
        self.assertWarns(UserWarning, find_nearest_examples, dataset, k, rule, class_col_name, lookup, min_max, classes,
                         label_type=my_vars.SAME_LABEL_AS_RULE, only_uncovered_neighbors=False)

    def test_find_neighbors_numeric_nominal(self):
        """Tests what happens if input has a numeric and a nominal feature"""
        df = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                           "C": [3, 2, 1, .5, 3, 2],
                           "Class": ["apple", "apple", "banana", "banana", "banana", "banana"]})
        class_col_name = "Class"
        lookup = \
            {
                "A":
                    {
                        'high': 2,
                        'low': 4,
                        my_vars.CONDITIONAL:
                            {
                                'high':
                                    Counter({
                                        'banana': 2
                                    }),
                                'low':
                                    Counter({
                                        'banana': 2,
                                        'apple': 2
                                    })
                            }
                    }
            }
        k = 4
        correct = None
        if k == 1:
            correct = df.iloc[[5]]
        elif k == 2:
            correct = df.iloc[[5, 2]]
        elif k == 3:
            correct = df.iloc[[5, 2, 3]]
        elif k >= 4:
            correct = df.iloc[[5, 2, 3, 4]]
        rule = pd.Series({"A": "high", "B": Bounds(lower=1, upper=1), "Class": "banana"})
        classes = ["apple", "banana"]
        min_max = pd.DataFrame({"A": {"min": 1, "max": 5}, "B": {"min": 1, "max": 11}})
        # Reset as other tests changed the content of the dictionary
        my_vars.closest_rule_per_example = {}
        neighbors, _, _ = find_nearest_examples(df, k, rule, class_col_name, lookup, min_max, classes,
                                                label_type=my_vars.SAME_LABEL_AS_RULE, only_uncovered_neighbors=False)
        if neighbors is not None:
            self.assertTrue(neighbors.shape[0] == k)
        self.assertTrue(neighbors.equals(correct))

    def test_find_neighbors_numeric_nominal_label_type(self):
        """Tests what happens if input has a numeric and a nominal feature and we vary label_type as parameter"""
        df = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                           "C": [3, 2, 1, .5, 3, 2],
                           "Class": ["apple", "apple", "banana", "banana", "banana", "banana"]})
        class_col_name = "Class"
        lookup = \
            {
                "A":
                    {
                        'high': 2,
                        'low': 4,
                        my_vars.CONDITIONAL:
                            {
                                'high':
                                    Counter({
                                        'banana': 2
                                    }),
                                'low':
                                    Counter({
                                        'banana': 2,
                                        'apple': 2
                                    })
                            }
                    }
            }
        k = 3
        rules = [
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=3, upper=3), "Class": "apple"},
                      name=0),
            pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=2, upper=2), "Class": "apple"},
                      name=1),
            pd.Series({"A": "high", "B": Bounds(lower=4, upper=4), "C": Bounds(lower=1, upper=1), "Class": "banana"},
                      name=2),
            pd.Series({"A": "low", "B": Bounds(lower=1.5, upper=1.5), "C": Bounds(lower=0.5, upper=0.5),
                       "Class": "banana"}, name=3),
            pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=0.5), "C": Bounds(lower=3, upper=3), "Class": "banana"},
                      name=4),
            pd.Series({"A": "high", "B": Bounds(lower=0.75, upper=0.75), "C": Bounds(lower=2, upper=2),
                       "Class": "banana"}, name=5)
        ]
        my_vars.all_rules = {0: rules[0], 1: rules[1], 2: rules[2], 3: rules[3], 4: rules[4], 5: rules[5]}
        my_vars.closest_rule_per_example = {}
        my_vars.closest_examples_per_rule = {}
        correct_all = df.iloc[[5, 2, 0]]
        correct_same = df.iloc[[5, 2, 3]]
        correct_opposite = df.iloc[[0, 1]]
        rule = pd.Series({"A": "high", "B": (1, 1), "Class": "banana"}, name=0)
        classes = ["apple", "banana"]
        min_max = pd.DataFrame({"A": {"min": 1, "max": 5}, "B": {"min": 1, "max": 11}})
        neighbors_all, _, _ = find_nearest_examples(df, k, rule, class_col_name, lookup, min_max, classes,
                                                    label_type=my_vars.ALL_LABELS, only_uncovered_neighbors=False)
        neighbors_same, _, _ = find_nearest_examples(df, k, rule, class_col_name, lookup, min_max, classes,
                                                     label_type=my_vars.SAME_LABEL_AS_RULE, only_uncovered_neighbors=
                                                     False)
        neighbors_opposite, _, _ = find_nearest_examples(df, k, rule, class_col_name, lookup, min_max, classes,
                                                         label_type=my_vars.OPPOSITE_LABEL_TO_RULE,
                                                         only_uncovered_neighbors=False)
        print(neighbors_all)
        print(neighbors_same)
        print(neighbors_opposite)
        self.assertTrue(neighbors_all.equals(correct_all))
        self.assertTrue(neighbors_same.equals(correct_same))
        self.assertTrue(neighbors_opposite.equals(correct_opposite))

    def test_find_neighbors_numeric_nominal_covered(self):
            """Tests what happens if input has a numeric and a nominal feature and some examples are already covered
            by the rule"""
            df = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                               "C": [3, 2, 1, .5, 3, 2],
                               "Class": ["apple", "apple", "banana", "banana", "banana", "banana"]})
            class_col_name = "Class"
            lookup = \
                {
                    "A":
                        {
                            'high': 2,
                            'low': 4,
                            my_vars.CONDITIONAL:
                                {
                                    'high':
                                        Counter({
                                            'banana': 2
                                        }),
                                    'low':
                                        Counter({
                                            'banana': 2,
                                            'apple': 2
                                        })
                                }
                        }
                }
            k = 4
            my_vars.closest_rule_per_example = {}
            correct = None
            if k == 1:
                correct = df.iloc[[5]]
            elif k == 2:
                correct = df.iloc[[5, 2]]
            elif k == 3:
                correct = df.iloc[[5, 2, 3]]
            elif k >= 4:
                # correct = df.iloc[[5, 2, 3, 4]]
                # Examples at indices 2 and 4 are already covered by the rule, so don't return them as neighbors
                my_vars.examples_covered_by_rule = {0: {2, 4}}
                correct = df.iloc[[5, 3]]
            my_vars.all_rules = {}
            rule = pd.Series({"A": "high", "B": Bounds(lower=1, upper=1), "Class": "banana"}, name=0)
            classes = ["apple", "banana"]
            min_max = pd.DataFrame({"A": {"min": 1, "max": 5}, "B": {"min": 1, "max": 11}})

            neighbors, _, _ = find_nearest_examples(df, k, rule, class_col_name, lookup, min_max, classes,
                                                    label_type=my_vars.SAME_LABEL_AS_RULE, only_uncovered_neighbors=
                                                    True)
            self.assertTrue(neighbors.equals(correct))

    def test_find_neighbors_numeric_nominal_stats(self):
            """Tests that global statistics are updated accordingly"""
            df = pd.DataFrame({"A": ["low", "low", "high", "low", "low", "high"], "B": [1, 1, 4, 1.5, 0.5, 0.75],
                               "C": [3, 2, 1, .5, 3, 2],
                               "Class": ["apple", "apple", "banana", "banana", "banana", "banana"]})
            class_col_name = "Class"
            lookup = \
                {
                    "A":
                        {
                            'high': 2,
                            'low': 4,
                            my_vars.CONDITIONAL:
                                {
                                    'high':
                                        Counter({
                                            'banana': 2
                                        }),
                                    'low':
                                        Counter({
                                            'banana': 2,
                                            'apple': 2
                                        })
                                }
                        }
                }
            rule = pd.Series({"A": "high", "B": (1, 1), "Class": "banana"}, name=0)
            my_vars.closest_rule_per_example = {
                0: Data(rule_id=1, dist=0.010000000000000002),
                1: Data(rule_id=0, dist=0.010000000000000002),
                2: Data(rule_id=5, dist=0.67015625),
                3: Data(rule_id=1, dist=0.038125),
                4: Data(rule_id=0, dist=0.015625),
                5: Data(rule_id=2, dist=0.67015625)}
            # Reset because other tests added data, so if you only run this test it would work, but not if other
            # tests are run prior to that
            my_vars.examples_covered_by_rule = {}
            my_vars.closest_examples_per_rule = {}
            my_vars.closest_examples_per_rule = {0: {1, 4}, 1: {0, 3}, 2: {5}, 5: {2}}
            rules = [
                pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=3, upper=3), "Class": "apple"},
                          name=0),
                pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=2, upper=2), "Class": "apple"},
                          name=1),
                pd.Series({"A": "high", "B": Bounds(lower=4, upper=4), "C": Bounds(lower=1, upper=1),
                           "Class": "banana"}, name=2),
                pd.Series({"A": "low", "B": Bounds(lower=1.5, upper=1.5), "C": Bounds(lower=0.5, upper=0.5),
                           "Class": "banana"}, name=3),
                pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=0.5), "C": Bounds(lower=3, upper=3),
                           "Class": "banana"}, name=4),
                pd.Series({"A": "high", "B": Bounds(lower=0.75, upper=0.75), "C": Bounds(lower=2, upper=2),
                           "Class": "banana"}, name=5)
            ]
            my_vars.all_rules = {0: rules[0], 1: rules[1], 2: rules[2], 3: rules[3], 4: rules[4], 5: rules[5]}
            # my_vars.all_rules = {0: rule}
            k = 4
            correct = df.iloc[[5, 2, 3, 4]]

            classes = ["apple", "banana"]
            my_vars.minority_class = "banana"
            min_max = pd.DataFrame({"A": {"min": 1, "max": 5}, "B": {"min": 1, "max": 11}})
            correct_covered = {}
            correct_examples_per_rule = {0: {1, 2, 4, 5}, 1: {0, 3}}
            correct_closest_rule_per_example = {
                0: Data(rule_id=1, dist=0.010000000000000002),
                1: Data(rule_id=0, dist=0.010000000000000002),
                2: Data(rule_id=0, dist=0.09),
                3: Data(rule_id=1, dist=0.038125),
                4: Data(rule_id=0, dist=0.015625),
                5: Data(rule_id=0, dist=0.0006250000000000001)}
            neighbors, _, _ = find_nearest_examples(df, k, rule, class_col_name, lookup, min_max, classes,
                                                    label_type=my_vars.SAME_LABEL_AS_RULE, only_uncovered_neighbors=
                                                    False)
            self.assertTrue(neighbors.equals(correct))
            self.assertTrue(correct_covered == my_vars.examples_covered_by_rule)
            self.assertTrue(correct_examples_per_rule == my_vars.closest_examples_per_rule)
            for example_id, (rule_id, dist) in correct_closest_rule_per_example.items():
                features = my_vars.all_rules[rule_id].size
                self.assertTrue(example_id in my_vars.closest_rule_per_example)
                other_id, other_dist = my_vars.closest_rule_per_example[example_id]
                other_features = my_vars.all_rules[other_id].size
                self.assertTrue(rule_id == other_id)
                self.assertTrue(features == other_features)
                self.assertTrue(abs(dist-other_dist) < 0.0001)

    def test_find_neighbors_numeric_nominal_covers(self):
            """Tests that the stats for a newly covered rule are updated (dist = 0)"""
            """Tests that global statistics are updated accordingly"""
            df = pd.DataFrame({"A": ["low", "low", "high", "high", "low", "high"], "B": [1, 1, 1, 1, 0.5, 0.75],
                               "C": [3, 2, 1, .5, 3, 2],
                               "Class": ["apple", "apple", "banana", "banana", "banana", "banana"]})
            class_col_name = "Class"
            lookup = \
                {
                    "A":
                        {
                            'high': 2,
                            'low': 4,
                            my_vars.CONDITIONAL:
                                {
                                    'high':
                                        Counter({
                                            'banana': 2
                                        }),
                                    'low':
                                        Counter({
                                            'banana': 2,
                                            'apple': 2
                                        })
                                }
                        }
                }
            rules = [
                pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=3, upper=3), "Class": "apple"},
                          name=0),
                pd.Series({"A": "low", "B": Bounds(lower=1, upper=1), "C": Bounds(lower=2, upper=2), "Class": "apple"},
                          name=1),
                pd.Series({"A": "high", "B": Bounds(lower=4, upper=4), "C": Bounds(lower=1, upper=1),
                           "Class": "banana"}, name=2),
                pd.Series({"A": "low", "B": Bounds(lower=1.5, upper=1.5), "C": Bounds(lower=0.5, upper=0.5),
                           "Class": "banana"}, name=3),
                pd.Series({"A": "low", "B": Bounds(lower=0.5, upper=0.5), "C": Bounds(lower=3, upper=3),
                           "Class": "banana"}, name=4),
                pd.Series({"A": "high", "B": Bounds(lower=0.75, upper=0.75), "C": Bounds(lower=2, upper=2),
                           "Class": "banana"}, name=5)
            ]
            my_vars.all_rules = {0: rules[0], 1: rules[1], 2: rules[2], 3: rules[3], 4: rules[4], 5: rules[5]}
            my_vars.closest_rule_per_example = {
                0: (1, 0.010000000000000002),
                1: (0, 0.010000000000000002),
                2: (5, 0.67015625),
                3: (1, 0.038125),
                4: (0, 0.015625),
                5: (2, 0.67015625)}
            my_vars.closest_examples_per_rule = {0: {1, 4}, 1: {0, 3}, 2: {5}, 5: {2}}
            k = 4
            correct = df.iloc[[2, 3, 5, 4]]
            rule = pd.Series({"A": "high", "B": (1, 1), "Class": "banana"}, name=0)
            classes = ["apple", "banana"]
            my_vars.minority_class = "banana"
            min_max = pd.DataFrame({"A": {"min": 1, "max": 5}, "B": {"min": 1, "max": 11}})
            # An example could be covered by multiple rules, so example 2 should be covered by rules 0 and 1 at the end
            my_vars.examples_covered_by_rule = {1: {2}}
            correct_covered = {0: {2, 3}, 1: {2}}
            correct_examples_per_rule = {0: {1, 2, 3, 4, 5}, 1: {0}}
            correct_closest_rule_per_example = {
                0: (1, 0.010000000000000002), 1: (0, 0.010000000000000002), 2: (0, 0.0), 3: (0, 0.0),
                4: (0, 0.015625), 5: (0, 0.0006250000000000001)}
            neighbors, _, _ = find_nearest_examples(df, k, rule, class_col_name, lookup, min_max, classes,
                                                    label_type=my_vars.SAME_LABEL_AS_RULE, only_uncovered_neighbors=
                                                    False)
            self.assertTrue(neighbors.equals(correct))
            self.assertTrue(correct_covered == my_vars.examples_covered_by_rule)
            self.assertTrue(correct_examples_per_rule == my_vars.closest_examples_per_rule)
            for example_id, (rule_id, dist) in correct_closest_rule_per_example.items():
                self.assertTrue(example_id in my_vars.closest_rule_per_example)
                other_id, other_dist = my_vars.closest_rule_per_example[example_id]
                self.assertTrue(rule_id == other_id)
                self.assertTrue(abs(dist - other_dist) < 0.0001)
