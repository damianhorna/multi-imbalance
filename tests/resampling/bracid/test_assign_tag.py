from unittest import TestCase
from collections import Counter

from multi_imbalance.resampling.bracid.bracid import BRACID, ExampleClass


class TestAssignTag(TestCase):
    """Tests assign_tag() in utils.py"""

    def test_assign_tag_safe_unanimously(self):
        """Tests if "safe" is assigned correctly when label is chosen unanimously"""
        bracid = BRACID()
        label = "a"
        labels = Counter(["a", "a", "a", "a"])
        tag = bracid.assign_tag(labels, label)
        self.assertEqual(tag, ExampleClass.SAFE)

    def test_assign_tag_safe(self):
        """Tests if "safe" is assigned correctly"""
        bracid = BRACID()
        label = "a"
        labels = Counter(["a", "b", "a", "c"])
        tag = bracid.assign_tag(labels, label)
        self.assertEqual(tag, ExampleClass.SAFE)

    def test_assign_tag_noisy(self):
        """Tests if "safe" is assigned correctly"""
        bracid = BRACID()
        label = "a"
        labels = Counter(["b", "b", "b", "b"])
        tag = bracid.assign_tag(labels, label)
        self.assertEqual(tag, ExampleClass.NOISY)

    def test_assign_tag_borderline_tie(self):
        """Tests if "borderline" is assigned correctly in case of ties"""
        bracid = BRACID()
        label = "a"
        labels = Counter(["a", "b", "a", "b"])
        tag = bracid.assign_tag(labels, label)
        self.assertEqual(tag, ExampleClass.BORDERLINE)

    def test_assign_tag_borderline(self):
        """Tests if "borderline" is assigned correctly"""
        bracid = BRACID()
        label = "a"
        labels = Counter(["a", "b", "b", "c"])
        tag = bracid.assign_tag(labels, label)
        self.assertEqual(tag, ExampleClass.BORDERLINE)