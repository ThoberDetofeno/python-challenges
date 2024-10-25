# https://edabit.com/challenge/XQwPPHE6ZSu4Er9ht
# Economical Numbers
# A number is Economical if the quantity of digits of its prime factorization (including exponents greater than 1) is equal to or lower than the digit quantity of the number itself.
# 
# Given an integer n, implement a function that returns a string:
# 
# "Equidigital" if the quantity of digits of the prime factorization (including exponents greater than 1) is equal to the quantity of digits of n;
# "Frugal" if the quantity of digits of the prime factorization (including exponents greater than 1) is lower than the quantity of digits of n;
# "Wasteful" if none of the two above conditions is true.
# Examples
# is_economical(14) ➞ "Equidigital"
# # The prime factorization of 14 (2 digits) is [2, 7] (2 digits)
# # Exponents equal to 1 are not counted
# 
# is_economical(125) ➞ "Frugal"
# # The prime factorization of 125 (3 digits) is [5^3] (2 digits)
# # Notice how exponents greater than 1 are counted
# 
# is_economical(1024) ➞ "Frugal"
# # The prime factorization of 1024 (4 digits) is [2^10] (3 digits)
# 
# is_economical(30) ➞ "Wasteful"
# # The prime factorization of 30 (2 digits) is [2, 3, 5] (3 digits)
# Notes
# Any given n will be a positive integer greater than 1.
# Remember to count also the exponents greater than 1 into the prime factorization: 2¹ = 2 (one digit), 2² = 22 (two digits), 2¹° = 210 (three digits)...

def is_economical(n):
    def nextPrimeNumber(prime):
        nextPrime = prime + 1
        while True:
            if isPrime(nextPrime):
                return nextPrime
            nextPrime += 1
    
    def isPrime(num):
        for i in range(2,num):
            if num % i == 0:
                return False
        return True
    
    def numberOfDigits(listOfDigits):
        num = 0
        for p in set(listOfDigits):
            num += len(str(p))
            if listOfDigits.count(p) > 1:
                num += len(str(listOfDigits.count(p)))
        return num
    #
    digits = []
    rest = n
    prime = 2
    while rest != 1:
        if rest%prime == 0:
            rest = int(rest/prime)
            digits.append(prime)
        else:
            prime = nextPrimeNumber(prime)
    #
    numDigits = numberOfDigits(digits)

    if len(str(n)) == numDigits:
        return "Equidigital"
    if len(str(n)) > numDigits:
        return "Frugal"
    return "Wasteful"
    

if __name__ == "__main__":
	print(is_economical(1267))