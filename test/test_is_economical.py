import unittest
from code_challenge.is_economical import is_economical

class test_is_economical(unittest.TestCase):

    def test_values(self):
        self.assertEqual(is_economical(14), "Equidigital", "Example #1")
        self.assertEqual(is_economical(125), "Frugal", "Example #2")
        self.assertEqual(is_economical(1024), "Frugal", "Example #3")
        self.assertEqual(is_economical(30), "Wasteful", "Example #4")
        self.assertEqual(is_economical(81), "Equidigital")
        self.assertEqual(is_economical(243), "Frugal")
        self.assertEqual(is_economical(5), "Equidigital")
        self.assertEqual(is_economical(6), "Wasteful")
        self.assertEqual(is_economical(1267), "Equidigital")
        self.assertEqual(is_economical(1701), "Frugal")
        self.assertEqual(is_economical(1267), "Equidigital")
        self.assertEqual(is_economical(12871), "Equidigital")
        self.assertEqual(is_economical(88632), "Wasteful")

if __name__ == "__main__":
    unittest.main()