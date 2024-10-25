import unittest
from code_challenge.freqCount import freq_count

class test_freqCount(unittest.TestCase):

    def test_values(self):
        self.assertEqual(freq_count([1, 1, 1, 1], 1), [[0, 4]])
        self.assertEqual(freq_count([1, 1, 2, 2], 1), [[0, 2]])
        self.assertEqual(freq_count([1, 1, 2, [1]], 1), [[0, 2], [1, 1]])
        self.assertEqual(freq_count([1, 1, 2, [[1]]], 1), [[0, 2], [1, 0], [2, 1]])
        self.assertEqual(freq_count([[[1]]], 1), [[0, 0], [1, 0], [2, 1]])
        self.assertEqual(freq_count([1, 4, 4, [1, 1, [1, 2, 1, 1]]], 1), [[0, 1], [1, 2], [2, 3]])
        self.assertEqual(freq_count([1, 5, 5, [5, [1, 2, 1, 1], 5, 5], 5, [5]], 5), [[0, 3], [1, 4], [2, 0]])
        self.assertEqual(freq_count([1, [2], 1, [[2]], 1, [[[2]]], 1, [[[[2]]]]], 2), [[0, 0], [1, 1], [2, 1], [3, 1], [4, 1]])

if __name__ == "__main__":
    unittest.main()