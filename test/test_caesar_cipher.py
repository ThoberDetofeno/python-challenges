# https://edabit.com/challenge/C45TKLcGxh8dnbgqM

import unittest
from code_challenge.caesar_cipher import caesar_cipher

class test_bar_chart(unittest.TestCase):

    def test_values(self):
        self.assertEqual(caesar_cipher("middle-Outz", 2), "okffng-Qwvb")
        self.assertEqual(caesar_cipher("Always-Look-on-the-Bright-Side-of-Life", 5), "Fqbfdx-Qttp-ts-ymj-Gwnlmy-Xnij-tk-Qnkj")
        self.assertEqual(caesar_cipher("A friend in need is a friend indeed", 20), "U zlcyhx ch hyyx cm u zlcyhx chxyyx")
        self.assertEqual(caesar_cipher("A Fool and His Money Are Soon Parted.", 27), "B Gppm boe Ijt Npofz Bsf Tppo Qbsufe.")
        self.assertEqual(caesar_cipher("One should not worry over things that have already happened and that cannot be changed.", 49), "Lkb pelria klq tloov lsbo qefkdp qexq exsb xiobxav exmmbkba xka qexq zxkklq yb zexkdba.")
        self.assertEqual(caesar_cipher("Back to Square One is a popular saying that means a person has to start over, similar to: back to the drawing board.", 126), "Xwyg pk Omqwna Kja eo w lklqhwn owuejc pdwp iawjo w lanokj dwo pk opwnp kran, oeiehwn pk: xwyg pk pda znwsejc xkwnz.")

if __name__ == "__main__":
    unittest.main()








