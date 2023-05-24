from unittest import TestCase
from collections import Counter

from scripts.vars import SAFE, NOISY, BORDERLINE
from scripts.bracid import assign_tag


class TestAssignTag(TestCase):
    """Tests assign_tag() in utils.py"""

    def test_assign_tag_safe_unanimously(self):
        """Tests if "safe" is assigned correctly when label is chosen unanimously"""
        label = "a"
        labels = Counter(["a", "a", "a", "a"])
        tag = assign_tag(labels, label)
        self.assertTrue(tag == SAFE)

    def test_assign_tag_safe(self):
        """Tests if "safe" is assigned correctly"""
        label = "a"
        labels = Counter(["a", "b", "a", "c"])
        tag = assign_tag(labels, label)
        self.assertTrue(tag == SAFE)

    def test_assign_tag_noisy(self):
        """Tests if "safe" is assigned correctly"""
        label = "a"
        labels = Counter(["b", "b", "b", "b"])
        tag = assign_tag(labels, label)
        self.assertTrue(tag == NOISY)

    def test_assign_tag_borderline_tie(self):
        """Tests if "borderline" is assigned correctly in case of ties"""
        label = "a"
        labels = Counter(["a", "b", "a", "b"])
        tag = assign_tag(labels, label)
        self.assertTrue(tag == BORDERLINE)

    def test_assign_tag_borderline(self):
        """Tests if "borderline" is assigned correctly"""
        label = "a"
        labels = Counter(["a", "b", "b", "c"])
        tag = assign_tag(labels, label)
        self.assertTrue(tag == BORDERLINE)
