# https://edabit.com/challenge/C45TKLcGxh8dnbgqM

# Julius Caesar protected his confidential information by encrypting it using a cipher. 
# Caesar's cipher (check Resources tab for more info) shifts each letter by a number of letters.
# If the shift takes you past the end of the alphabet, just rotate back to the front of the alphabet.
# In the case of a rotation by 3, w, x, y and z would map to z, a, b and c.
# 
# Create a function that takes a string s (text to be encrypted) and an integer k (the rotation factor). It should return an encrypted string.
# 
# Examples
# caesar_cipher("middle-Outz", 2) ➞ "okffng-Qwvb"
# 
#   m -> o
#   i -> k
#   d -> f
#   d -> f
#   l -> n
#   e -> g
#   -    -
#   O -> Q
#   u -> w
#   t -> v
#   z -> b
# 
# caesar_cipher("Always-Look-on-the-Bright-Side-of-Life", 5)
# ➞ "Fqbfdx-Qttp-ts-ymj-Gwnlmy-Xnij-tk-Qnkj"
# 
# caesar_cipher("A friend in need is a friend indeed", 20)
# ➞ "U zlcyhx ch hyyx cm u zlcyhx chxyyx"

def caesar_cipher(s, k):
    newString = ''
    k = k - (k//26 * 26)
    for w in s:
        nrAscii = ord(w)
        newAscii = nrAscii + k
        if (nrAscii > 64) & (nrAscii < 91): # Uppercase  
            nrAscii = newAscii if not(newAscii > 90) else newAscii - 26
        elif (nrAscii > 96) & (nrAscii < 123): # Lowercase 
            nrAscii = newAscii if not(newAscii > 122) else newAscii - 26
        newString += chr(nrAscii)
    return newString

if __name__ == "__main__":
    print(caesar_cipher("One should not worry over things that have already happened and that cannot be changed.", 49))