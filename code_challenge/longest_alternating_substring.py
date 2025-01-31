# https://edabit.com/challenge/RB6iWFrCd6rXWH3vi
# Given a string of digits, return the longest substring with alternating odd/even or even/odd digits.
# If two or more substrings have the same length, return the substring that occurs first.
#
# Examples
# longest_substring("225424272163254474441338664823") ➞ "272163254"
# substrings = 254, 272163254, 474, 41, 38, 23
#
# longest_substring("594127169973391692147228678476") ➞ "16921472"
# substrings = 94127, 169, 16921472, 678, 476
#
# longest_substring("721449827599186159274227324466") ➞ "7214"
# substrings = 7214, 498, 27, 18, 61, 9274, 27, 32
# 7214 and 9274 have same length, but 7214 occurs first.

def longest_substring(digits):
    subStrings = []
    oldD = newSubString = digits[0]
    for d in digits[1:]:
        if (int(d)%2 == int(oldD)%2):
            subStrings.append(newSubString)
            newSubString = d
        else:
            newSubString += d
        oldD = d
    subStrings.append(newSubString)
    return max(subStrings, key=len)

if __name__ == "__main__":
    longest_substring("594127169973391692147228678476")

