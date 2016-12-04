#QUESTION 1:
def length(string):
  str_amount= 0 
  for i in string:
    str_amount+= ord(i)
  return str_amount

def question1(s, t):
  length_s = len(s)
  length_t = len(t)

  beginning = 0
  ending = length_t

  while ending <= length_s and t!='':
    new_char = s[beginning:ending]
    long_amount=0
    if length(new_char) == length(t):
      return True
    else:
      beginning +=1
      ending+=1
  return False
  
## Test cases:
print question1('udacity', 'ad')
#True

print question1('','a')
#False

print question1('hello', '')
#False

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
#123321

##################################
#QUESTION3:

def find_min(input_val):
  min_value = min(input_val)
  # index_val = input_val[min_value]
  return input_val.index(min_value)

def valid(self):
  self= False

def question3(self):
  #list of all_nodes (i.e. ['A','B','C'])
  all_nodes = self.keys()

  #tuples per node:
  new_g = {}
  for el in all_nodes:
    
    tuples =  self[el]
    nodes =[]
    edges=[]
    for i in range(0, len(tuples)):
      nodes.append(tuples[i][0])
      edges.append(tuples[i][1])

    min_node = nodes[find_min(edges)]
    
    new_g[el] = [(min_node, edges[find_min(edges)])]

  return new_g


#Test case:

G=({'A': [('B', 2), ('C', 8)],
 'B': [('A', 2), ('C', 5)], 
 'C': [('B', 5), ('A', 8)]})

print question3(G)
#{'A': [('B', 2)], 'C': [('B', 5)], 'B': [('A', 2)]}

S = ({'A': [('C', 3), ('B', 7)],
      'B': [('A', 7), ('D', 1)],
      'D': [('B', 1), ('C', 7)],
      'C': [('A', 3), ('D', 7)]})

print question3(S)
#{'A': [('C', 3)], 'C': [('A', 3)], 'B': [('D', 1)], 'D': [('B', 1)]}

T = ({'A': [('C', 3), ('B', 7), ('D', 1)],
      'B': [('A', 7), ('D', 1)],
      'D': [('B', 1), ('C', 7)],
      'C': [('A', 3), ('D', 7)],
      'D': [('A',1)]})

print question3(T)
# {'A': [('D', 1)], 'C': [('A', 3)], 'B': [('D', 1)], 'D': [('A', 1)]}

########################

# QUESTION4:

def question4(T, root, n1, n2):
  #odd cases:
  if n1 == n2:
    return n1
  if (root == None) or (n1==None) or (n2==None):
    return None

  min_val = min(n1, n2)
  max_val = max(n1, n2)

  value = root

  while (value != None):
    if (value >= min_val) and (value <= max_val):
      return value 
    elif value > max_val:
      sub_list = T[value][:value+1]
      value = [i for i,x in enumerate(sub_list) if x == 1][0]    
    elif value < min_val:
      sub_list = T[value][value:]
      value = [i for i,x in enumerate(sub_list) if x == 1][0]


#test cases:
T =[[0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 0, 0, 0, 1],
    [0, 0, 0, 0, 0]]

'''
     3
    / \
   0   4 
    \
     1 

'''
print question4(T,3,1,4)
#3

print question4(T,None, 1,2)
#None

print question4(T,3, 1,1)
#1


T1 = [[0, 0, 1],
      [0, 0, 0],
      [0, 0, 0]]

'''
   0
    \
     2

'''

print question4(T1,0,2,0)
#0

###############################

# QUESTION5:

class Node(object):
  def __init__(self, data):
    self.data = data
    self.next = None


def question5(ll, m):
  if ll == None:
    return None
  else:
    temp = ll
    for i in range(m):
      if temp == None:
        return None
      temp = temp.next
    if temp == None:
      return ll.data
    else:
      return question5(ll.next, m)    



A = Node(5)
B = Node(1)
C = Node(0)
D = Node(7)
E = Node(8)
F = Node(2)
G = Node(3)
H = Node(4)

A.next = B
B.next = C
C.next = D
D.next = E
E.next = F
F.next = G
G.next = H

#test cases:
print question5(A,3)
#2

print question5(None, 1)
#none

print question5(A, 9)
#None

