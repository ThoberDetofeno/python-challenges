import unittest
from code_challenge.distance_to_nearest_vowel import distance_to_nearest_vowel

class test_distance_to_nearest_vowel(unittest.TestCase):

    def test_values(self):
        self.assertEqual(distance_to_nearest_vowel("aaaaa"), [0, 0, 0, 0, 0])
        self.assertEqual(distance_to_nearest_vowel("bba"), [2, 1, 0])
        self.assertEqual(distance_to_nearest_vowel("abbb"), [0, 1, 2, 3])
        self.assertEqual(distance_to_nearest_vowel("abab"), [0, 1, 0, 1])
        self.assertEqual(distance_to_nearest_vowel("babbb"), [1, 0, 1, 2, 3])
        self.assertEqual(distance_to_nearest_vowel("baaab"), [1, 0, 0, 0, 1])
        self.assertEqual(distance_to_nearest_vowel("abcdabcd"), [0, 1, 2, 1, 0, 1, 2, 3])
        self.assertEqual(distance_to_nearest_vowel("abbaaaaba"), [0, 1, 1, 0, 0, 0, 0, 1, 0])
        self.assertEqual(distance_to_nearest_vowel("treesandflowers"), [2, 1, 0, 0, 1, 0, 1, 2, 2, 1, 0, 1, 0, 1, 2])
        self.assertEqual(distance_to_nearest_vowel("pokerface"), [1, 0, 1, 0, 1, 1, 0, 1, 0])
        self.assertEqual(distance_to_nearest_vowel("beautiful"), [1, 0, 0, 0, 1, 0, 1, 0, 1])
        self.assertEqual(distance_to_nearest_vowel("rythmandblues"), [5, 4, 3, 2, 1, 0, 1, 2, 2, 1, 0, 0, 1])
        self.assertEqual(distance_to_nearest_vowel("shopper"), [2, 1, 0, 1, 1, 0, 1])
        self.assertEqual(distance_to_nearest_vowel("singingintherain"), [1, 0, 1, 1, 0, 1, 1, 0, 1, 2, 1, 0, 1, 0, 0, 1])
        self.assertEqual(distance_to_nearest_vowel("sugarandspice"), [1, 0, 1, 0, 1, 0, 1, 2, 2, 1, 0, 1, 0])
        self.assertEqual(distance_to_nearest_vowel("totally"), [1, 0, 1, 0, 1, 2, 3])

if __name__ == "__main__":
    unittest.main()