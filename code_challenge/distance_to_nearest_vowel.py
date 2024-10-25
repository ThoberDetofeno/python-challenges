# https://edabit.com/challenge/jWHkKc2pYmgobRL8R
# Distance to Nearest Vowel
# Write a function that takes in a string and for each character, returns the distance to the nearest vowel in the string. If the character is a vowel itself, return 0.
# 
# Examples
# distance_to_nearest_vowel("aaaaa") ➞ [0, 0, 0, 0, 0]
# 
# distance_to_nearest_vowel("babbb") ➞ [1, 0, 1, 2, 3]
# 
# distance_to_nearest_vowel("abcdabcd") ➞ [0, 1, 2, 1, 0, 1, 2, 3]
# 
# distance_to_nearest_vowel("shopper") ➞ [2, 1, 0, 1, 1, 0, 1]
# Notes
# All input strings will contain at least one vowel.
# Strings will be lowercased.
# Vowels are: a, e, i, o, u.

def distance_to_nearest_vowel(txt):
    vowels = ['a', 'e', 'i', 'o', 'u']
    result = []
    for i, w in enumerate(txt):
        wUp = wDown = w
        iUp = iDown = i
        distance = 0
        foundDistance = False
        while not(foundDistance):
            if (wUp in vowels) or (wDown in vowels):
                foundDistance = True
                result.append(distance)
            else:
                iUp += 1
                iDown -= 1
                if iUp < len(txt):
                    wUp = txt[iUp]
                if iDown >= 0:
                    wDown = txt[iDown]
                distance += 1
        
    return result

if __name__ == '__main__':
    print(distance_to_nearest_vowel("singingintherain"))