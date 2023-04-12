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

	def greedy( self,time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		found_tour = False
		count = 0
		bssf = None
		start_time = time.time()

		#The current node that we start at, I started at node A every time
		curr_node = 0
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

	def branchAndBound( self, time_allowance=60.0 ):
		pass



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
		bssf = cities[:]
		num_sols = 0
		len_cities = len(cities)
		improved = True
		time_start = time.time()

		# loop until no improvement is made
		while improved and time.time() - time_start < time_allowance:
			improved = False
			# loop through all edges and try to swap them
			for i in range(1, len_cities):
				for j in range(i + 1, len_cities):
					if j - i == 1:
						continue
					distance_original = distance(cities[i-1], cities[i]) + distance(cities[j], cities[j-1])
					distance_new =		distance(cities[i-1], cities[j]) + distance(cities[i], cities[j-1])

					# if new distance is shorter, swap the edges
					if distance_new < distance_original:
						cities[i:j] = reversed(cities[i:j])
						improved = True
					else:
						#if new path is longer, decide whether to accept it or not
						if random.random() < math.exp(-(distance_new - distance_original)):
							cities[i:j] = reversed(cities[i:j])
							improved = True

			# if improved, update the best solution so far
			if improved:
				bssf = cities[:]
				num_sols += 1

		# Set final results and return
		results['cost'] = 999
		results['time'] = time.time() - time_start
		results['count'] = num_sols
		results['soln'] = TSPSolution(bssf)
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results

