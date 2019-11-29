import collections
from collections import defaultdict
import queue
import random
import copy
import time
import sys

class LinkedList():
	def __init__(self):
		self.start = None

	def add_to_end(self, n):
		if not self.start:
			self.start = Node1(n)
		else:
			curr = self.start
			while curr.next:
				curr = curr.next
			curr.next = Node1(n)

	def insert_at_front(self, n):
		if not self.start:
			self.start = Node1(n)
		else:
			temp = Node1(n)
			temp.next = self.start
			self.start = temp 

	def traverse(self):
		curr = self.start
		while curr:
			print(curr.val, end=" ")
			curr = curr.next 

	def insert_after_another(self, n, n1):
		if not self.start:
			print("List is empty")

		to_insert = Node1(n)
		curr = self.start 

		while curr:
			if curr.val == n1:
				break
			curr = curr.next
		if not curr:
			print("No n1 node found")
		else:
			to_insert.next = curr.next
			curr.next = to_insert

	def remove(self, n):
		pass


class Node1:
	def __init__(self, n, next_node=None):
		self.val = n
		self.next = next_node


class Node:
	def __init__(self, val, left=None, right=None):
		self.val = val
		self.left = left
		self.right = right

class BST:
	num_nodes: int
	root: Node

	def __init__(self):
		self.num_nodes = 0
		self.root = None

	def add(self, val):
		if not self.root:
			self.root = Node(val)
			self.num_nodes += 1
		else:
			curr = self.root
			if self.look(curr, val):
				self.num_nodes += 1

	def look(self, curr, val) -> bool:
		if val > curr.val:
			if not curr.right:
				curr.right = Node(val)
				return True
			else:
				return self.look(curr.right, val)
		elif val < curr.val:
			if not curr.left:
				curr.left = Node(val)
				return True
			else:
				return self.look(curr.left, val)
		else:
			return False


def main():
	#num_in_common([1, 2, 10], [2, 10, 11])
	root = Node(7)
	root.left = Node(11)
	#root.left.right = Node(11)
	root.right = Node(8)
	root.right.right = Node(9)
	root.right.right.right = Node(10)

	nodes = [3, 5, 2, -4, 56, 3, 4]
	tree = BST()
	
	for n in nodes:
		tree.add(n)

	llist = LinkedList()
	llist.add_to_end(1)
	llist.add_to_end(3)
	llist.insert_at_front(0)
	llist.insert_after_another(2, 1)
	
	#print(longest_ascending_subsequence(list(range(10, 0, -1))))
	print(eight_queens())


#Google
#########################################################

def isValid(s):
	#can have curly, brackets, and paranthesis
	#have counters for each type
	expect = {')': '(', '}': '{', ']' : '['}
	q = list()

	for c in s:
		if c not in expect:
			q.append(c)
		else:
			try:
				popped = q.pop()
				if popped != expect[c]:
					return False
			except IndexError:
				return False
	
	return not q


#########################################################

#DROPBOX
#########################################################
#10-15 11-12 12-13 13-14 14-15 15-16
def abc_alt(n, nb, nc):
	if not n:
		return 1
	result = abc_alt(n-1, nb, nc)
	if nb>0:
		result += abc_alt(n-1, nb-1, nc)
	if nc > 0:
		result += abc_alt(n-1, nb, nc-1)

	return result


#########################################################
#CHAPTER 10

#2.3 ########################################################

def detect_corruption(llist):
	#questions: possible that the list is not corrupt?
	#what if >1 corruption?
	#A -> A

	#loop thru the nodes, adding each nde to a set if not there
	# if there, then corrupt

	curr = llist.start
	seen = set()
	seen.append(curr)

	while curr:
		curr = curr.next
		if curr in seen:
			return curr

	return None

#########################################################

#2.3 ########################################################

def delete_middle(mid):
	#1->2->3->4
	if not mid or not mid.next:
		return

	curr.val = curr.next.val
	curr.next = curr.next.next
	
#########################################################

#2.2 ########################################################

def return_kth_to_last(l, k):
	#4 -> 5 -> 6 -> 7 -> \

	first = l.start
	if not first:
		return None

	for i in range(k):
		first = first.next 
		if not first:
			return None

	second = l.start

	while first.next:
		first = first.next
		second = second.next

	return second.val


	
########################################################

#FB ########################################################

def find_all_cubes():
	cont = list()

	for num in range(10000):
		sum_ = 0
		num_ = num
		while num:
			last = num % 10
			sum_ += last ** 3
			num //= 10
		if sum_ == num_:
			cont.append(num_)

	return(cont)


########################################################
def get_power_of_10(remainder, divisor): #0.3333
    
    for p in range(1, 50):
        remainder1 = remainder #0.3333
        print('rem1', remainder1)
        whole1 = remainder1 * (10 ** p) // divisor #1
        print('wjole1', whole1)
        temp1 = whole1 * divisor
        print('temp1', temp1)
        remainder2 = remainder1  * (10 ** p) - temp1 #0.333
        print('rem2', remainder2)
        whole2 = remainder2 * (10 ** p) // divisor #1
        print('whole2', whole2)
        temp2 = whole2 * divisor
        print('temp2', temp2)
        if temp1 == temp2:
            return p

    return None



########################################################
def longest_ascending_subsequence(arr):
	#1 2 3 1 1 3 6 2 7 8 9 23 4
	max_len = 1
	counter = max_len

	if not arr:
		return None

	prev = arr[0]
	for elem in arr[1:]:

		if elem > prev:
			counter += 1
		else:
			if counter > max_len:
				max_len = counter
			counter = 1

		prev = elem

	return max(counter, max_len)


########################################################

########################################################
def biggest_element(node):
	if not node:
		return 0

	return max(biggest_element(node.left), node.val, biggest_element(node.right))

########################################################

def make_list_out_of_tree(root):
	nodes = list()
	traverse_tree(root, nodes)

	if not nodes:
		return None
	start = Node1(nodes[0])
	curr = start
	for n in nodes[1:]:
		curr.next = Node1(n)
		curr = curr.next

	return start

def traverse_tree(node, arr):
	if not node:
		return 

	traverse_tree(node.left, arr)
	arr.append(node.val)
	traverse_tree(node.right, arr)


"""
def make_list_out_of_tree(root):
	llist_cont = [None]
	start_cont = [None]
	build_list(root, llist_cont, start_cont)
	return start_cont[0]

	#traverse the tree in order

def build_list(node, lnode, start_cont):
	if not node:
		return 

	build_list(node.left, lnode, start_cont)

	if not lnode[0]:
		#start
		lnode[0] = Node1(node.val)
		start_cont[0] = lnode[0]
	else:
		#all other cases
		lnode[0].next = Node1(node.val)
		lnode[0] = lnode[0].next

	build_list(node.right, lnode, start_cont)

"""

def print_tree_by_level(root):
	dic = defaultdict(int)
	curr = 0

	get_levels(root, dic, curr)

	for level in dic:
		print(f"Level {level}: {dic[level]}")

def get_levels(node, dic, curr):
	if not node:
		return

	dic[curr].append(node.val)
	get_levels(node.left, dic, curr+1)
	get_levels(node.right, dic, curr+1)

########################################################


########################################################
def sorted_arr_to_BST(arr):
	root = None

	return sorted_arr_to_BST_helper(root, arr, 0, len(arr) - 1)

def sorted_arr_to_BST_helper(node, arr, start, end):
	if start > end:
		return None

	mid = (start + end) // 2
	val = arr[mid]

	node = Node(val)

	node.left = sorted_arr_to_BST_helper(node.left, arr, start, mid-1)
	node.right = sorted_arr_to_BST_helper(node.right, arr, mid+1, end)

	return node

########################################################

########################################################

#cs221 ########################################################
"""
def is_palindrome(text):
        start = 0
        end = len(text) - 1

        while (start < end):
            if text[start] != text[end]:
                return False
            start += 1
            end -= 1
        return True

def computeLongestPalindromeLength1(text):


def compute_helper(text, dic):
	length = len(text)

	if text in dic:
		return dic[text]

	if is_palindrome(text):
		dic[text] = length
		return length

	half = length // 2

	if text[0] == text[-1]:
		dic[text] = 2 + compute_helper(text[1:-1], dic)
		return dic[text]

	text1 = text[1:]
	text2 = text[:-1]
	dic[text1] = compute_helper(text1, dic)
	dic[text2] = compute_helper(text2, dic)

	return max(dic[text1], dic[text2])






def computeLongestPalindromeLength(text):
 
    A palindrome is a string that is equal to its reverse (e.g., 'ana').
    Compute the length of the longest palindrome that can be obtained by deleting
    letters from |text|.
    For example: the longest palindrome in 'animal' is 'ama'.
    Your algorithm should run in O(len(text)^2) time.
    You should first define a recurrence before you start coding.

    # BEGIN_YOUR_CODE (our solution is 19 lines of code, but don't worry if you deviate from this)

    def is_palindrome(text):
        start = 0
        end = len(text) - 1

        while (start < end):
            if text[start] != text[end]:
                return False
            start += 1
            end -= 1
        return True

    def palindrome(text, cont, seen):
    	text_len = len(text)
    	#for efficiency
    	if cont[0] >= text_len:
            return 
    	print(text)

    	#memo #####
    	if text in seen:
    		return 

    	seen.add(text)

    	##########
    	if is_palindrome(text):
        	print("ITS A PALINDROME")
        	if text_len > cont[0]:
        		cont[0] = len(text)
        	return

    	half = len(text) // 2
    	for i in range(half):
		    first = i
		    last = text_len - (i+1)
		    if text[first] == text[last]:
		        continue
		    text1 = text[:first] + text[first+1:]
		    text2 = text[:last] + text[last+1:]

		    palindrome(text1, cont, seen)
		    palindrome(text2, cont, seen)

    cont = [0]
    palindrome(text, cont, set())
    return cont.pop()

"""

########################################################

#CHAPTER 10

#10.11 ########################################################

def sort_peaks_and_valleys(arr):
	"""
	in the array {5, 8, 6, 2, 3, 4, 6}, {8, 6} are peaks and {5, 2} are valleys. 
	Given an array of integers, sort the array into an alternating sequence of peaks and valleys.
	O(nlogn) + O(n)
	"""

	arr = sort_1011(arr)
	for first in range(0, len(arr)-1, 2):
		print(first)
		swap_(arr, first, first+1)

	return arr

def swap_(arr, first, second):
	arr[first], arr[second] = arr[second], arr[first]

def sort_1011(arr):
	length = len(arr)
	if length <= 1:
		return arr

	left = sort_1011(arr[:length // 2])
	right = sort_1011(arr[length//2:])

	new_arr = []

	left_i = 0
	right_i = 0

	while left_i < len(left) and right_i < len(right):
		left_elem = left[left_i]
		right_elem = right[right_i]
		if left_elem < right_elem:
			new_arr.append(left_elem) 
			left_i += 1
		else:
			new_arr.append(right_elem)
			right_i += 1

	if left_i == len(left):
		new_arr += right[right_i:]
	else:
		new_arr += left[left_i:]

	return new_arr

########################################################

# 10.9 ########################################################

#implement find_sorted_hard
def find_sorted_matrix(matrix, elem):
	M = len(matrix)
	col = len(matrix[0]) - 1
	row = 0

	while col >= 0 and row <= M-1:
		if matrix[row][col] == elem:
			return (row, col)
		elif matrix[row][col] > elem:
			col -= 1
		else:
			row += 1

	return None

#WRONG
def sorted_matrix_search_advanced(matrix, elem):
	#O(log M log N)
	M = len(matrix)
	N = len(matrix[0])
	top_row = matrix[M-1]
	left_col = [x for x in matrix[N-1]]

	row_nums = find_potentials(0, N-1, top_row, elem)
	col_nums = find_potentials(0, M-1, left_col, elem)

	if row_nums is int:
		return (0, row_nums)
	if col_nums is int:
		return (col_nums, 0)

	print(row_nums, col_nums)

	for row in row_nums:
		if row > N-1:
			continue
			
		for col in col_nums:
			if col > M-1:
				continue

			if matrix[row][col] == elem:
				return (row, col)

	return None

#WRONG
def find_potentials(start, stop, arr, elem):
	if start > stop:
		return (start, stop)

	mid = (start + stop) // 2

	if arr[mid] == elem:
		return mid
	elif elem < arr[mid]:
		return find_potentials(start, mid-1, arr, elem)
	else:
		return find_potentials(mid+1, stop, arr, elem)


def sorted_matrix_search_simple(matrix, elem):
	"""Given an M x N matrix in which each row and each column is sorted in 
	ascending order, write a method to find an element.
	edge cases = empty matrix, Nx1, 1xM
	m = [[3, 5, 9], 
		[4, 6, 11], 
		[10, 11, 12] ]
	"""

	M = len(matrix)
	N = len(matrix[0])

	for row in matrix:
		index = binary_search(0, N-1, row, elem)
		if index:
			return index

	#rigth away soln - loop thru each row and binary search - O(M log N)

def binary_search(start, stop, arr, elem):
	if start > end:
		return None

	mid = (start + stop) // 2

	if arr[mid] == elem:
		return mid
	elif elem < arr[mid]:
		return binary_search(start, mid-1, arr, elem)
	else:
		return binary_search(mid+1, stop, arr, elem)
#########################################################

# 10.5 ########################################################

def sparse_search(word, arr, start, end):
	"""
	Sparse Search: Given a sorted array of strings that is interspersed with empty strings, 
	write a method to find the location of a given string.

	Example: Input: ball, {"at", "", "", "ate", "ball", "", "", "dad", "", ""}
	Output: 4
	"""
	if end < start or not arr:
		return -1

	med = (start + end) // 2
	curr = arr[med]
	if curr == word:
		return med

	if not curr: #searching both ways
		i = sparse_search(word, arr, start, med-1)
		return i if i != -1 else sparse_search(word, arr, med+1, end)
	elif word < curr:
		return sparse_search(word, arr, start, med-1)
	else: #word > curr
		return sparse_search(word, arr, med+1, end)


#########################################################


# 10.4 ########################################################

def search_sorted_no_size(x, listy, start, end):
	#ideas: treat int_max as the last index, whenever there's -1, search the left part
	# do a binary search to find the last index, then do regular binary search

	#run time: really O(1) beacause we know the array can't be bigger than a certain value
	#another approach: calculate the length by starting at 1 and multiplying by 2
	if end < start:
		return None

	med = (start + end) // 2

	if listy[med] == x:
		return med
	elif listy[med] == -1 or x < listy[med]:
		return search_sorted_no_size(x, listy, start, med-1) #double check indexes, edge cases
	else: #x > listy[med]
		return search_sorted_no_size(x, listy, med + 1, end)

#########################################################

# 10.3 ########################################################
#slow soln
def search_in_rotated_arr(val, arr):
	if not arr:
		return None

	pivot = find_pivot(arr)
	new_arr = arr[pivot:] + arr[:pivot]
	length = len(new_arr)
	index = binary_search(val, new_arr, 0, length - 1)

	return (index + pivot) % length  

def find_pivot(arr):
	prev = 0
	for i in range(1, len(arr)):
		if arr[prev] > arr[i]:
			return i
		prev = i

	return 0
#DOESNT WORK
def find_pivot_fast(arr, start, end, prev=None):
	if start > end or end < start:
		return None

	if start == end:
		return start

	# 14, 15, 2, 3, 4, 5, 6, 7
	med = (start + end) // 2
	curr = arr[med]

	if prev:
		if curr < prev:
			return find_pivot_fast(arr, start, med-1, curr)
		elif curr > prev:
			return find_pivot_fast(arr, med+1, end, curr)
	else:
		one = find_pivot_fast(arr, start, med-1, curr)
		if one:
			return one
		two = find_pivot_fast(arr, med+1, end, curr)
		return two



def binary_search(value, arr, start, end):
	med = (start + end) // 2
	curr = arr[med]

	#edge case
	if start == end and value != curr:
		return None

	if value == curr:
		return med
	elif value < curr:
		return binary_search(value, arr, start, med - 1)
	else:
		return binary_search(value, arr, med + 1, end)


#########################################################

# 10.2 ########################################################
#question: how many characters can strings consist of?
#assume: only lower case alphabet
def sort_anagrams(arr):
	length = len(arr)
	if length <= 2:
		return arr

	dic = defaultdict(list)
	arr_of_arrays = []
	sorted_arr = []

	for word in arr:
		#address: what if strings repeat? can create a hashmap
		arr_of_arrays.append("".join(sorted(word)))

	for index in range(length):
		dic[arr_of_arrays[index]].append(index)

	for entry in dic:
		sorted_arr += [arr[x] for x in dic[entry]]

	return sorted_arr

"""
def build_array(word):
	arr = [0] * 26
	for c in word:
		arr[ord(c) - ord('a')] += 1
	return "".join([str(x) for x in arr])
"""
#########################################################

# quick-sort ########################################################
# time: O(n log n)
# space: O(log n)
def quick_sort(arr):
	length = len(arr)
	if not arr or length == 1:
		return arr

	pivot_index = random.randint(0, length-1)
	pivot = arr[pivot_index]

	left_part = []
	right_part = []

	for i in range(length):
		if i == pivot_index:
			continue
		elem = arr[i]
		left_part.append(elem) if elem < pivot else right_part.append(elem)

	return quick_sort(left_part) + [pivot] + quick_sort(right_part)


#########################################################

# merge-sort ########################################################
#ex: [5, 3, 6, 10, -5]
#	[5, 3, 6]	[10, -5]
#	[5, 3] [6] 	[10] [-5]
# 	[5] [3] [6] [10] [-5]
# 	[3, 5] [6] [-5, 10]
#	[3, 5, 6] [-5, 10]
# 	[-5, 3, 5, 6, 10]

def merge_sort(arr):
	#base case
	if not arr or len(arr) == 1:
		return arr

	#split
	length = len(arr)
	arr_left = arr[:length // 2]
	arr_right = arr[length // 2:]

	#recurse
	arr_left = merge_sort(arr_left)
	arr_right = merge_sort(arr_right)

	sorted_arr = merge(arr_left, arr_right)
	return sorted_arr

def merge(arr_left, arr_right):
	#merge
	left_length = len(arr_left)
	right_length = len(arr_right)
	left_iter = 0
	right_iter = 0

	sorted_arr = []
	while left_iter < left_length and right_iter < right_length:
		left_elem = arr_left[left_iter]
		right_elem = arr_right[right_iter]
		if left_elem < right_elem:
			sorted_arr.append(left_elem)
			left_iter += 1
		else:
			sorted_arr.append(right_elem)
			right_iter += 1

	if left_iter == left_length:
		sorted_arr += arr_right[right_iter:]
	else:
		sorted_arr += arr_left[left_iter:]


	return sorted_arr


#########################################################

#CHAPTER 1

# 1.9 ########################################################

def string_rotation(tup):
	"""
	apple, pleap

	are two congruent strings rotations of each other?
	"""
	s1 = tup[0]
	s2 = tup[1]

	if len(s1) != len(s2):
		return False

	s1_doubled = s1 + s1
	return s2 in s1_doubled

def is_substring(a, b):
	return b in a

########################################################

# 1.5 ########################################################
def one_away(stra, strb):
	"""
	There are three types of edits that can be performed on strings: insert a character, 
	remove a character, or replace a character. Given two strings, write a function to check 
	if they are one edit (or zero edits) away.

	time: O(n), where n is the length of the shorter string
	"""

	lena = len(stra)
	lenb = len(strb)

	#quick base case
	if -2 <= lena-lenb >= 2:
		return False


	main_word, tested_word, len_r = (stra, strb, lenb) if lena > lenb else (strb, stra, lena)

	#shorter length or mutual length
	for i in range(len_r):
		if main_word[i] != tested_word[i]:
			return True if main_word[i+1:] == tested_word[i:] or main_word[i+1:] == tested_word[i+1:] else False
	return True


#########################################################

# 1.4 ########################################################

def palindrome_perm(s):
	"""
	time: O(n)
	also could have a cool soln with bit masks!
	"""
	num_odds = 0
	dic = defaultdict(int)

	for c in s:
		dic[c] += -1 if dic[c] else 1

	for key in dic:
		if key != ' ' and dic[key]:
			num_odds += 1
	return num_odds <= 1


#########################################################

# 1.3 ########################################################
#better for long strings
def URLify(s):
	t1 = time.time()
	ans = "".join(list(map(lambda x: '%20' if x == ' ' else x, s)))
	t2 = time.time()
	print(t2-t1)
	return ans

def URLify_slow(s):
	t1 = time.time()
	new_s = ""
	for c in s:
		new_s += '%20' if c == ' ' else c

	t2 = time.time()
	print(t2-t1)
	return new_s

#########################################################

# 1.2 ########################################################

def is_permutation_opt(stra, strb):
	"""
	time: O(n)
	space: O(n)
	"""
	if len(stra) != len(strb):
		return False

	a = defaultdict(int)
	b = defaultdict(int)

	for c in stra:
		a[c] += 1
	for c in strb:
		b[c] += 1

	return a == b

def is_permutation(stra, strb):
	"""
	time: O(n log n)
	space: O(n)
	"""
	if len(stra) != len(strb):
		return False

	return [c for c in stra].sort() == [c for c in strb].sort()


#########################################################

# 1.1 ########################################################
def is_unique(s):
	"""
	check if the string has only unique chars w/o data structures
	time: O(n^2)
	space: O(1)
	"""
	count = 0
	for c in s:
		count += 1
		if c in s[count:]:
			return False
	return True
#########################################################

#CHAPTER 8

# 8.14 ########################################################

def count_eval_main(inputt: str, result: bool) -> int:
	"""
	 Given a boolean expression consisting of the symbols 0 (false), 1 (true), & (AND), 
	 | (OR), and ^ (XOR), and a desired boolean result value result, implement a function 
	 to count the number of ways of parenthesizing the expression such that it evaluates to result.
	"""
	memo = dict()
	return count_eval(inputt, result, memo)

def count_eval(inputt: str, result: bool, memo) -> int:
	if not str:
		return 0
	if len(str) == 1:
		return int( bool(inputt) == result )
	if inputt + str(result) in memo:
		return memo[inputt]

	num_ways = 0
	length = len(inputt)
	for i in range(1, length, 2):
		c = inputt[i]
		left = inputt[:i]
		right = inputt[i+1:]

		left_true = count_eval(left, True, memo)
		left_false = count_eval(left, False, memo)
		right_true = count_eval(right, True, memo)
		right_false = count_eval(right, False, memo)
		total_true = (left_true + left_false) * (right_true + right_false)

		ways_true = 0
		if c == '&':
			#to be true, both must be true
			ways_true = left_true * right_true 
		elif c == '^':
			#one true and one false
			ways_true = left_true * right_false + left_false * right_true
		elif c == '|':
			#both arent false
			ways_true = total_true - left_false * right_false

		subways = ways_true if result else total_true - ways_true
		ways += subways

	memo[inputt + str(result)] = ways
	return ways

########################################################

# 8.12 ########################################################

def eight_queens():
	board = [[0 for c in range(8)] for j in range(8)]
	#prev_pos = set()
	return eight_queens_helper(8, board)

def eight_queens_helper(num_left, board, prev_r=0, prev_c=0):
	if not num_left:
		print_board(board)
		print()
		return 1

	ways = 0

	for r in range(prev_r, 8):
		for c in range(prev_c, 8):
			if not board[r][c]:
				prev_board = copy.deepcopy(board)
				board[r][c] = 'Q'
				fill(r, c, board)
				ways += eight_queens_helper(num_left-1, board, r, c)
				board = prev_board
		prev_c = 0

	return ways

def fill(r, c, board):
	#horizontal
	new_arr = 8 * [1]
	new_arr[c] = 'Q'
	board[r] = new_arr

	#vertical
	for i in range(8):
		i %= 8
		if i != r:
			board[i][c] = 1

	#diagonals
	temp_r_neg, temp_r_pos = r, r
	temp_c_neg, temp_c_pos = c, c

	while temp_r_neg > 0 and temp_c_neg > 0:
		temp_r_neg -= 1
		temp_c_neg -= 1
		board[temp_r_neg][temp_c_neg] = 1

	while temp_r_pos > 0 and temp_c_pos < 7:
		temp_r_pos -= 1
		temp_c_pos += 1
		board[temp_r_pos][temp_c_pos] = 1


	temp_r_neg, temp_r_pos = r, r
	temp_c_neg, temp_c_pos = c, c

	while temp_r_neg < 7 and temp_c_neg < 7:
		temp_r_neg += 1
		temp_c_neg += 1
		board[temp_r_neg][temp_c_neg] = 1

	while temp_r_pos < 7 and temp_c_pos > 0:
		temp_r_pos += 1
		temp_c_pos -= 1
		board[temp_r_pos][temp_c_pos] = 1


def print_board(board):
	for row in board:
		for val in row:
			print(val, end=' ') 
		print()






########################################################

# 8.11 ########################################################

def coins(n):
	return coins_helper(n, 0)

def coins_helper(n, prev, currency=[1, 5, 10, 25]):
	if n < 0:
		return 0

	if not n:
		return 1
	
	ways = 0
	for i in range(prev, len(currency)):
		ways += coins_helper(n-currency[i], i)

	return ways


#########################################################

# 8.9 ########################################################
def parens(n):
	return parens_helper(n)

def parens_helper(n, curr=''):
	if not n:
		print(curr)
		return 

	s1 = '(' + curr + ')'
	s2 = '()' + curr
	s3 = curr + '()'

	parens_helper(n-1, s1)

	if s2 != s1:
		parens_helper(n-1, s2)
	if s3 != s1 and s3 != s2:
		parens_helper(n-1, s3)

#########################################################

# 8.8 ########################################################
def perm_w_dups(s):
	container = list()
	if not s:
		pass
	elif len(s) == 1:
		container.append(s)
	else:
		c = s[0]
		for word in perm_w_dups(s[1:]):
			for i in range(len(word)+1):
				if i < len(word) and word[i] == c:
					continue
				container.append(word[0:i] + c + word[i:])
	return container

########################################################


# 8.7 ########################################################

def perm_no_dups_optimized(s):
	container = list()
	if not s:
		pass
	elif len(s) == 1:
		container.append(s)
	else:
		c = s[0]
		for word in perm_no_dups_optimized(s[1:]):
			for i in range(len(word)+1):
				container.append(word[0:i] + c + word[i:])

	return container



def perm_no_dups(s):
	#time complexity: O(N^3)
	container = list()
	if not s:
		return container

	container.append(s[0])
	for c in s[1:]:
		temp = perm_no_dups_helper(c, container)
		container = temp

	return container

def perm_no_dups_helper(c, container):
	new_temp = list()
	for word in container:
		for i in range(len(word) + 1):
			new_temp.append(word[0:i] + c + word[i:])

	return new_temp



########################################################
# 8.6 ########################################################

def move_hanoi(n, t1, t2, t3):
	if not (descending(t1) and descending(t2) and descending(t3)):
		raise ValueError("Didn't follow the rules") 

	if not n:
		return
	move_hanoi(n - 1, t1, t3, t2)
	t3.append(t1.pop())
	move_hanoi(n - 1, t2, t1, t3)
	#FINISH LATER

def descending(tower):
	l = len(tower)
	for i in range(l-1):
		if tower[i] < tower[i+1]:
			return False
	return True


########################################################

# 8.5 ########################################################
def recursive_multiply(a, b):
	return recursive_multiply_2(max(a, b), min(a, b))

def recursive_multiply_2(bigger, smaller):
	if not smaller:
		return 0
	elif smaller == 1:
		return bigger
	elif smaller & 1: #odd
		return bigger + recursive_multiply_2(bigger, smaller - 1)
	else: #even
		half_prod = recursive_multiply_2(bigger, smaller >> 1)
		return half_prod + half_prod




########################################################

# 8.4 ########################################################
#1, 2, 3

def get_power_sets(S):
	empty = frozenset()
	big_set = set()
	big_set.add(empty)


	for elem in S:
		temp_set = set()
		for mem in big_set:
			new_set = set(mem)
			new_set.add(elem)
			temp_set.add(frozenset(new_set))
		big_set |= temp_set
			
	#test cases: empty set
	return big_set

########################################################

# 8.3 ########################################################

def get_magic_index(arr):
	#[-4, 1, 4, 6, 9]
	#find magix index (arr[i] == i) if it exists

	return find_index(arr, 0, len(arr))
	

def find_index(arr, start, end):
	if start == end:
		return None

	i = (start + end) // 2
	if arr[i] == i:
		return i
	else:
		if arr[i] > i:
			return find_index(arr, 0, i)
		else:
			return find_index(arr, i+1, end)


########################################################

# 8.1 ########################################################
def triple_steps(n):
	"""
	A child is running up a staircase with n steps and can hop either 1 step, 
	2 steps, or 3 steps at a time. Implement a method to count how many possible 
	ways the child can run up the stairs.
	"""
	memo = [None] * (n+1)

	return triple_steps_memo(n, memo)

def triple_steps_memo(n, memo):
	if n == 0:
		return 1
	elif n < 0:
		return 0
	else:
		if memo[n] == None:
			memo[n] = (triple_steps_memo(n-1, memo) + 
				triple_steps_memo(n-2, memo) + triple_steps_memo(n-3, memo))

		return memo[n]

def triple_steps_bruteforce(n):
	if n == 0:
		return 1
	elif n < 0:
		return 0
	else:
		return triple_steps(n-1) + triple_steps(n-2) + triple_steps(n-3)

########################################################

#fib ########################################################
#runtime, space - O(n)
def fibonacci(n):
	memo = (n+1) * [None]
	print(fib_memo(n, memo))

def fib_memo(n, memo):
	if n == 1 or n == 0:
		return n
	if not memo[n]: 
		memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)

	return memo[n]

########################################################

#CHAPTER 4

#4.11 ########################################################

#fast soln
def get_random_node_optimized(root):
	#generate one random num which will indicate the random
	#node in the in-order traversal

	num = 3 #some generated random in

	if not root:
		return None

	return find_Nth_node(root, num)

def find_Nth_node(node, num):
	if not node:
		left_size = 0
	else:
		left_size = node.left_size


	if num < left_size:
		return find_Nth_node(node.left, num)
	elif num == left_size:
		return node
	else:
		return find_Nth_node(node.right, num - left_size - 1)

#slow solution
def get_random_node(root):
	"""
	returns a random node from the tree. 
	All nodes should be equally likely to be chosen
	"""

	#idea -- inorder traverse the tree and build a list with its
	#nodes. Time/space = O(n)
	arr = list()
	build_inorder_nodelist(root, arr)
	selected = random.choice(arr)
	print(selected)
	print(selected.val)

def build_inorder_nodelist(node, arr):
	if not node:
		return
	else:
		build_inorder_nodelist(node.left, arr)
		arr.append(node)
		build_inorder_nodelist(node.right, arr)

########################################################

#4.5 ########################################################

def validate_BST(root):
	"""
	Implement a function to check if a binary tree is a binary search tree.
	"""
	#BST if right child is greater, left child is less, and each subtree is BST
	#possibilies: can be null, can 
	#idea: traverse inorder - O(n), check if sorted - O(n)

	prev = float('-inf')
	return traverse_and_check(root, [prev])


def traverse_and_check(node, prev_cont) -> bool:
	if not node: 
		return True
	else:
		left_worked = traverse_and_check(node.left, prev_cont)
		if not left_worked or node.val < prev_cont[0]:
			return False
		prev_cont[0] = node.val
		right_worked = traverse_and_check(node.right, prev_cont)

		return left_worked and right_worked


########################################################

#4.4 ########################################################

#efficient soln
def is_balanced_efficient(root):
	return check_height_efficient(root) != -10

def check_height_efficient(root):
	if not root:
		return -1

	left_height = check_height_efficient(root.left)
	if left_height == -10:
		return -10

	right_height = check_height_efficient(root.right)
	if right_height == -10:
		return -10

	if -1 <= left_height - right_height <= 1:
		return 1 + max(left_height, right_height)
	else:
		return -10


#inefficient soln
def is_balanced(node):
	"""
	Implement a function to check if a binary tree is balanced. For the purposes of this 
	question, a balanced tree is defined to be a tree such that the heights of the two 
	subtrees of any node never differ by more than one.
	"""
	
	#tree is balanced if its subtrees' height don't differ by more than 1
	#and both subtrees are balanced
	#prob will need a helper height func

	if not node:
		return True
	#false
	return ( -1 <= get_height(node.left) - get_height(node.right) <= 1 and 
		is_balanced(node.left) and is_balanced(node.right) )

def get_height(node):
	if not node:
		return -1
	else:
		return 1 + max(get_height(node.left), get_height(node.right))
########################################################

#4.3 ########################################################

def get_depth_lists(root):
	"""
	Given a binary tree, design an algorithm which creates a linked list of all the nodes
at each depth (e.g., if you have a tree with depth D, you'll have D linked lists)
	"""
	depth_lists = defaultdict(list)

	#need a helper function
	#pass in a counter for each level
	#pass in the dict

	build_lists(root, depth_lists, 0)
	return depth_lists


def build_lists(node, depth_lists, depth):
	if not node:
		return

	depth_lists[depth].append(node.val)
	build_lists(node.left, depth_lists, depth+1)
	build_lists(node.right, depth_lists, depth+1)

########################################################

#4.10 ########################################################

def check_subtree(tree1, tree2) -> bool:
	#ideas: traverse a tree, at each node check if it's the right subtree
	if not tree2:
		return True
	if not tree1:
		return False

	#return check_subtree_helper(tree1, tree2)
	return check_subtree_with_preorder(tree1, tree2)

def check_subtree_helper(node1, node2):
	if not node1:
		return False

	if is_same_tree(node1, node2):
		return True

	return check_subtree_helper(node1.left, node2) or check_subtree_helper(node1.right, node2)

def is_same_tree(node_n, node2):
	if not node_n and not node2:
		return True
	elif not node_n or not node2:
		return False
	elif node_n.val != node2.val:
		return False
	else:
		return is_same_tree(node_n.left, node2.left) and is_same_tree(node_n.right, node2.right)

#another way -- substrings

def check_subtree_with_preorder(tree1, tree2):
	arr1 = list()
	arr2 = list()
	get_preorder(tree1, arr1)
	get_preorder(tree2, arr2)
	str1 = "".join(arr1)
	str2 = "".join(arr2)

	return str2 in str1

def get_preorder(node, arr):
	if not node:
		arr.append('N')
	else:
		arr.append(str(node.val))
		get_preorder(node.left, arr)
		get_preorder(node.right, arr)


#4.2 ########################################################

def build_tree(array):
	""" Given a sorted (increasing order) array with unique integer elements, write an algo­
	rithm to create a binary search tree with minimal height. """

	#example: 1 2 3 4 5 6 
	#insert the element in the middle
	#call the same function on the part to the right and to the left
	if not array:
		return None
	else:
		index = len(array) // 2
		node = Node(array[index])
		node.left = build_tree(array[:index])
		node.right = build_tree(array[index + 1:])

	return node


def checks():
	tree1 = build_tree([])
	traverse_inorder(tree1)

	tree2 = build_tree([1, 2])
	traverse_inorder(tree2)

	tree3 = build_tree([1, 2, 3, 5, 6])


########################################################


#4.1 ########################################################
def path_exists_dfs(currnode, finalnode, path, visited):
	#dfs
	visited.insert(currnode)

	if currnode.val == finalnode.val:
		print(path)
		return True

	for node in currnode.neighbors:
		if node in visited:
			continue

		path.append(node)
		if path_exists_dfs(node, finalnode, path):
			return True
		path.pop()

	return False

def path_exists_bfs(startnode, finalnode):
	paths = dict()
	q = queue.Queue()
	q.push(startnode)

	while not q.empty():
		currnode = q.get()

		for node in currnode.neighbors:
			q.put(node)
			paths[node] = currnode

			if node == finalnode:
				print(construct_path_bfs(startnode, finalnode, paths))
				return True

	return False

def construct_path_bfs(startnode, finalnode, paths):
	path = list()
	curr = finalnode

	while curr != startnode:
		path.append(curr)
		curr = path[curr]

	path.append(curr)
	path = path.reverse()
	return path
########################################################
	
def traverse_inorder(node):
	if not node:
		return
	else:
		traverse_inorder(node.left)
		print(node.val)
		traverse_inorder(node.right)

def traverse_preorder(node):
	if not node:
		return
	else:
		print(node.val)
		traverse_preorder(node.left)
		traverse_preorder(node.right)
		

def traverse_postorder(node):
	if not node:
		return
	else:
		traverse_postorder(node.left)
		traverse_postorder(node.right)
		print(node.val)
		
		

def num_in_common(A, B):
	#arrays are sorted
	num = 0
	start_index = 0
	for a in A:
		for i in range(start_index, len(B)):
			start_index = i
			if a == B[i]:
				num += 1
				break	
			elif a < B[i]:
				break
	print(num)


def permute(s):
	#a -  a
	#ab - ab, ba
	#abc - cab, acb, abc, cba, bca, bac
	if len(s) <= 1:
		return [s]
	else:
		ch = s[0]
		results = list()
		for perm in permute(s[1:]):
			for loc in range(len(perm)):
				results.append(perm[:loc] + ch + perm[loc:])
			results.append(perm + ch)
		return results


def find_permutations_of_s_in_b(s, b):
	s_ = [c for c in s]

	for i in range(len(b) - len(s) + 1):
		for j in len(s):
			if b[i + j] in s_:
				s_.remove(b[i+j])
			else:
				break
		if not s:
			print(i)




def sum_of_cubes(n):
	#O(N^2)
	dic = defaultdict(list)

	for a in range(1, n+1):
		for b in range(1, n+1):
			result = a ** 3 + b ** 3
			dic[result] += [(a, b)]

	for result in dic:
		for pair1 in dic[result]:
			for pair2 in dic[result]:
				print(pair1, pair2)


	"""
	#O(N^3)
	n = 1000
	for a in range(1, n+1):
		for b in range(1, n+1):
			for c in range(1, n+1):
				d = (a ** 3 + b ** 3 - c ** 3) ** (1/3)
				if a ** 3 + b ** 3 == c ** 3 + d ** 3:
					print(a, b, c, d)
	"""
if __name__ == '__main__':
	main()