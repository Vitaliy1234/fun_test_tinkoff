import unittest
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             os.path.pardir))

from bot import preprocess_sent


class TestClass(unittest.TestCase):
    def test_preprocess_sent_1(self):
        test_sent = 'Гомоморфный образ группы изоморфен фактор группе по ядру гомоморфизма'
        expected = np.array([[ 4.,  1.,  0.,  4.,  1.,  0.,  0.,  3.,  2.,  2.,  1.,  0., 36.,
         2., 65., 16.,  8.,  0.,  1.,  2.,  4.,  0.,  0.,  0.,  0.,  0.,
         0.,  2.,  0.,  0.,  0.,  1.,  0.]])

        actual = preprocess_sent(test_sent)

        self.assertTrue(np.equal(expected.all(), actual.all()))

    def test_preprocess_sent_2(self):
        test_sent = 'Гомоморфный образ , группы - изоморфен = фактор ijiji группе по ядру ... гомоморфизма'
        expected = np.array([[4., 1., 0., 4., 1., 0., 0., 3., 2., 2., 1., 0., 36.,
                              2., 65., 16., 8., 0., 1., 2., 4., 0., 0., 0., 0., 0.,
                              0., 2., 0., 0., 0., 1., 0.]])

        actual = preprocess_sent(test_sent)

        self.assertTrue(np.equal(expected.all(), actual.all()))

    def test_preprocess_sent_3(self):
        test_sent = 'Гомоморфный образ , группы - изоморф3ен = ijфijактор ijiji группе по ядру ... гомоморфизма'
        expected = np.array([[4., 1., 0., 4., 1., 0., 0., 3., 2., 2., 1., 0., 36.,
                              2., 65., 16., 8., 0., 1., 2., 4., 0., 0., 0., 0., 0.,
                              0., 2., 0., 0., 0., 1., 0.]])

        actual = preprocess_sent(test_sent)

        self.assertTrue(np.equal(expected.all(), actual.all()))