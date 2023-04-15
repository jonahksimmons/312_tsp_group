#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT6':
	from PyQt6.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))


import time
import numpy as np
from TSPClasses import *
import heapq
import itertools


def distance(city1, city2):
	return math.sqrt((city2._x - city1._x) ** 2 + (city2._y - city1._y) ** 2)


class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution,
		time spent to find solution, number of permutations tried during search, the
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

	def greedy( self,time_allowance=60.0, starting_node=0 ) -> dict:
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		found_tour = False
		count = 0
		bssf = None
		start_time = time.time()

		#The current node that we start at
		# this allows us to run it multiple times and get different solutions
		curr_node = starting_node
		visited_nodes = [curr_node]
		curr_city = visited_nodes[-1]

		while len(visited_nodes) < ncities:
			min_dist = np.inf
			city_index = None
			for neighborCity in range(ncities):
				#Checks and calculates the minimum distance, and makes sure it doesn't find any nodes that have been visited before
				if cities[curr_city].costTo(cities[neighborCity]) < min_dist and visited_nodes.count(neighborCity) == 0:
					city_index = neighborCity
					min_dist = cities[curr_city].costTo(cities[neighborCity])
			#If there were no paths less than infinity
			if min_dist == np.inf:
				curr_city = visited_nodes[-2]
			#Add the city with the smallest val to the route and set the curr_city to it
			else:
				visited_nodes.append(city_index)
				curr_city = visited_nodes[-1]


		route = []
		#Adds all the cities to the route and then creates the bssf from it.
		if len(visited_nodes) == ncities:
			for i in range(ncities):
				route.append(cities[visited_nodes[i]])
			bssf = TSPSolution(route)
			found_tour = True
			count += 1
		else:
			count = "No Solution"

		end_time = time.time()
		results['cost'] = bssf.cost if found_tour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None

		return results



	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints:
		max queue size, total number of states created, and number of pruned states.</returns>
	'''

	# Jacob: My branch and bound starts to hit the time limit starting at about 20 cities.
	#          When it hits time limit, it still returns a solution, but probably not optimal.
	# 		   IDK, if all we need is data to compare, then I guess it's good enough
	def branchAndBound(self, time_allowance=60.0):
		# Initialize variables and start timer
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		count = 1
		max_queue = 0
		num_pruned = 0
		total_states = 0
		start_time = time.time()

		# Create initial cost matrix and heap
		temp_bssf = self.defaultRandomTour().get('soln')
		self.bssf = self.CostObj(temp_bssf.cost, [], [], temp_bssf.route, 0)
		self.init_cost = np.zeros((ncities, ncities))
		for i in range(ncities):
			for j in range(ncities):
				if i == j:
					self.init_cost[i, j] = np.inf
					continue
				self.init_cost[i, j] = cities[i].costTo(cities[j])
		first_obj = self._reduceMatrix(self.CostObj(0, self.init_cost, [], [], 0), 0, True)
		self.init_cost = first_obj._matrix.copy()
		self.min_heap = [first_obj]
		heapq.heapify(self.min_heap)

		# Loop until time limit is reached or heap is empty
		while time.time() - start_time < time_allowance:
			if len(self.min_heap) == 0:
				break
			if len(self.min_heap) > max_queue:
				max_queue = len(self.min_heap)
			# Pop object with lowest cost from heap
			cur_obj = heapq.heappop(self.min_heap)
			# Loop through unvisited cities
			for i in range(len(cities)):
				if i in cur_obj._path:
					continue
				# Update total states explored
				total_states += 1
				# Reduce matrix and calculate new cost
				red_obj = self._reduceMatrix(cur_obj, i)
				if red_obj._cost < self.bssf._cost:
					# Add new object to heap if it has a lower cost than the current bssf
					heapq.heappush(self.min_heap, red_obj)
					# Update bssf if new object visits all cities
					if len(red_obj._path) == ncities:
						count += 1
						self.bssf = self.CostObj(red_obj._cost, red_obj._matrix.copy(), red_obj._path.copy(),
												 red_obj._city_path.copy(), red_obj._index)
				else:
					# Update number of pruned states
					num_pruned += 1
		end_time = time.time()

		# Record results
		results['cost'] = self.bssf._cost if self.bssf is not None else np.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = TSPSolution(self.bssf._city_path)
		results['max'] = max_queue
		results['total'] = total_states
		results['pruned'] = num_pruned

		# Return results
		return results

	# Function for reducing cost matrix
	# BnB Helper function
	def _reduceMatrix(self, cur_obj, dest, first=False):
		# Create a copy of the matrix and cost
		matrix_copy = cur_obj._matrix.copy()
		cost_copy = cur_obj._matrix.copy()

		# Set indices for the start and end cities
		start_city = cur_obj._index
		end_city = dest

		# Initialize reduction_cost to 0
		reduction_cost = 0

		if not first:
			# Set the distance between the start and end cities to infinity
			matrix_copy[start_city, end_city] = np.inf
			matrix_copy[end_city, start_city] = np.inf

			# Set the row and column of the start city to infinity
			matrix_copy[start_city] = np.inf
			matrix_copy[:, end_city] = np.inf

			# Calculate the reduction cost based on the current cost and the cost between the start and end cities
			reduction_cost += cur_obj._cost
			reduction_cost += cost_copy[cur_obj._index, dest]

		# Reduce the rows and columns of the matrix
		for row in range(len(matrix_copy)):
			# Find the minimum value in the row
			min_row = matrix_copy[row].min()
			if min_row != np.inf:
				# Subtract the minimum value from the row
				reduction_cost += min_row
				matrix_copy[row] = matrix_copy[row] - min_row

		# Reduce the columns of the matrix
		for col in range(len(matrix_copy)):
			# Find the minimum value in the column
			min_col = matrix_copy[:, col].min()
			if min_col != np.inf:
				# Subtract the minimum value from the column
				reduction_cost += min_col
				matrix_copy[:, col] = matrix_copy[:, col] - min_col

		# Create copies of the path and city path lists, and append the destination city to them
		path_copy = cur_obj._path.copy()
		path_copy.append(dest)

		city_path_copy = cur_obj._city_path.copy()
		city_path_copy.append(self._scenario.getCities()[dest])

		# Create and return a new CostObj instance
		return self.CostObj(reduction_cost, matrix_copy, path_copy, city_path_copy, dest)

	# Object to organize the data needed
	# BnB Helper object
	class CostObj:

		def __init__(self, cost, matrix, path, city_path, index):
			self._matrix = matrix
			self._cost = cost
			self._path = path
			self._city_path = city_path
			self._index = index

		def __lt__(self, cmp):
			if len(self._path) > len(cmp._path):
				return True
			if len(self._path) == len(cmp._path) and self._cost < cmp._cost:
				return True
			return False

	# BnB Helper function
	def get_cost(self, solution) -> float:
		return solution["cost"]

	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found during search, the
		best solution found.  You may use the other three field however you like.
		algorithm</returns>
	'''

	# Two-Opt Algorithm
	def fancy(self, time_allowance=60.0):
		# This object holds the results of the algorithm solution
		results = {}

		cities = self._scenario.getCities()

		# 2-opt starts with a fast, poor solution and improves it.
		# In this case, it's the best of 10 different greedy solutions.
		# This means that the final cost <= initial cost
		bssf: dict = min(
					[self.greedy(time_allowance, i) for i in range(min(10, len(cities)))], 
				key=self.get_cost)
		print("initial cost:", bssf["cost"])

		num_sols = 0
		improved = True
		time_start = time.time()

		# loop until no improvement is made
		while improved and time.time() - time_start < time_allowance:
			improved = False
			route = bssf['soln'].route

			# loop through all edges and try to swap them
			# Time complexity: O(n^2)
			for i in range(len(route)-2):  # TODO: unsure if these bounds are right
				for j in range(i + 1, len(route)-1):  # TODO: same here
					# the cost of two existing paths
					# compared to the cost if the destinations were swapped
					distance_original = distance(route[i], route[i+1]) + distance(route[j], route[j+1])
					distance_new =		distance(route[i], route[j+1]) + distance(route[j], route[i+1])

					# if new path is longer, decide whether to accept it or not
					# tempararily removing this for consistency
					random_chance = random.random() < math.exp(-(distance_new - distance_original))

					# if new distance is shorter, swap the edges
					if distance_new < distance_original:
						route[i:j] = reversed(route[i:j])
						improved = True

			# if improved, update the best solution so far
			if improved:
				# This might be the part where the results aren't set propperly
				new_solution = TSPSolution(route)
				bssf["soln"] = new_solution
				bssf["cost"] = new_solution.cost
				num_sols += 1

		# TODO: I think I messed up some part of the result so it doesn't display
		# Set final results and return
		print("Final solution", bssf["cost"])
		results['cost'] = bssf["cost"]
		results['time'] = time.time() - time_start
		results['count'] = num_sols
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None  # 2-opt does not prune
		return results
	


