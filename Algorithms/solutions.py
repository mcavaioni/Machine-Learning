# #QUESTION 1:
# def length(string):
#   str_amount= 0 
#   for i in string:
#     str_amount+= ord(i)
#   return str_amount

# def question1(s, t):
#   length_s = len(s)
#   length_t = len(t)

#   beginning = 0
#   ending = length_t

#   while ending <= length_s and t!='':
#     new_char = s[beginning:ending]
#     long_amount=0
#     if length(new_char) == length(t):
#       return True
#     else:
#       beginning +=1
#       ending+=1
#   return False
  
# ## Test cases:
# print question1('udacity', 'ad')
# #True

# print question1('','a')
# #False

# print question1('hello', '')
# #False

######################################
#QUESTION2:
from itertools import groupby
from operator import itemgetter
def question2(a):

  start = 0
  length = len(a)

  low=0
  high=0
  max_l=1
  for i in range(1, length):
    low = i - 1
    high = i
    while low >= 0 and high < length and a[low] == a[high]:
        if high - low + 1 > max_l:
            start = low
            max_l = high - low + 1
        low -= 1
        high += 1

    low = i-1
    high = i+1
    while low>=0 and high < length and a[low]==a[high]:
      if high-low +1 >max_l:
        start = low
        max_l= high-low+1
      low-=1
      high+=1
  return a[start:(start+max_l)]    

print question2('aaabba')  
#abba

print question2('ABCDEFCBA')
#A

print question2('ab123321')
               
