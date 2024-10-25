import unittest
from code_challenge.josephus import josephus

class test_bar_chart(unittest.TestCase):

    def test_values(self):
        self.assertEqual(josephus(41, 3), 31)
        self.assertEqual(josephus(14, 2), 13)
        self.assertEqual(josephus(35, 11), 18)
        self.assertEqual(josephus(20, 1), 20)
        self.assertEqual(josephus(15, 15), 4)

if __name__ == "__main__":
    unittest.main()