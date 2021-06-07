#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 17:56:47 2021
@author: frederic
    
class problem with     
An open-pit mine is a grid represented with a 2D or 3D numpy array. 
The first coordinates are surface locations.
In the 2D case, the coordinates are (x,z).
In the 3D case, the coordinates are (x,y,z).
The last coordinate 'z' points down.
    
A state indicates for each surface location  how many cells 
have been dug in this pit column.
For a 3D mine, a surface location is represented with a tuple (x,y).
For a 2D mine, a surface location is represented with a tuple (x,).
Two surface cells are neighbours if they share a common border point.
That is, for a 3D mine, a surface cell has 8 surface neighbours.
An action is represented by the surface location where the dig takes place.
"""
import numpy as np
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import itertools

import functools

from numbers import Number

import search


def my_team():
    '''    Return the list of the team members of this assignment submission 
    as a list    of triplet of the form (student_number, first_name, last_name)        '''
    return [(8799997, 'James', 'Tonkin'), (10512977, 'Lydia', 'Chen'), (10098101, 'Linh', 'Vu')]

def convert_to_tuple(a):
    '''
    Convert the parameter 'a' into a nested tuple of the same shape as 'a'.
    The parameter 'a' must be array-like. That is, its elements are indexed.
    Parameters
    ----------
    a : flat array or an array of arrays
    Returns
    -------
    the conversion of 'a' into a tuple or a tuple of tuples
    '''
    if isinstance(a, Number):
        return a
    if len(a) == 0:
        return ()
    # 'a' is non empty tuple
    if isinstance(a[0], Number):
        # 'a' is a flat list
        return tuple(a)
    else:
        # 'a' must be a nested list with 2 levels (a matrix)
        return tuple(tuple(r) for r in a)


def convert_to_list(a):
    '''
    Convert the array-like parameter 'a' into a nested list of the same 
    shape as 'a'.
    Parameters
    ----------
    a : flat array or array of arrays
    Returns
    -------
    the conversion of 'a' into a list or a list of lists
    '''
    if isinstance(a, Number):
        return a
    if len(a) == 0:
        return []
    # 'a' is non empty tuple
    if isinstance(a[0], Number):
        # 'a' is a flat list
        return list(a)
    else:
        # 'a' must be a nested list with 2 levels (a matrix)
        return [list(r) for r in a]


class Mine(search.Problem):
    '''
    Mine represent an open mine problem defined by a grid of cells 
    of various values. The grid is called 'underground'. It can be
    a 2D or 3D array.
    The z direction is pointing down, the x and y directions are surface
    directions.
    An instance of a Mine is characterized by 
    - self.underground : the ndarray that contains the values of the grid cells
    - self.dig_tolerance : the maximum depth difference allowed between 
                           adjacent columns 
    Other attributes:
        self.len_x, self.len_y, self.len_z : int : underground.shape
        self.cumsum_mine : float array : cumulative sums of the columns of the 
                                         mine
    A state has the same dimension as the surface of the mine.
    If the mine is 2D, the state is 1D.
    If the mine is 3D, the state is 2D.
    state[loc] is zero if digging has not started at location loc.
    More generally, state[loc] is the z-index of the first cell that has
    not been dug in column loc. This number is also the number of cells that
    have been dugged in the column.
    States must be tuple-based.
    '''

    def __init__(self, underground, dig_tolerance=1):
        '''
        Constructor
        Initialize the attributes
        self.underground, self.dig_tolerance, self.len_x, self.len_y, self.len_z,
        self.cumsum_mine, and self.initial
        The state self.initial is a filled with zeros.
        Parameters
        ----------
        underground : np.array
            2D or 3D. Each element of the array contains 
            the profit value of the corresponding cell.
        dig_tolerance : int
             Mine attribute (see class header comment)
        Returns
        -------
        None.
        '''
        # super().__init__() # call to parent class constructor not needed

        self.underground = underground
        # self.underground  should be considered as a 'read-only' variable!
        self.dig_tolerance = dig_tolerance
        # Make sure the array is 2d or 3d
        assert underground.ndim in (2, 3)

        # Assign the length of the mine into x, y, z; create the initial state
        if underground.ndim == 2:
            # 2d Mine
            self.len_x, self.len_z = underground.shape
            self.len_y = None
            # Initial state 0's as nothing has been mined
            self.initial = convert_to_tuple(np.zeros((self.len_x,), int))
            # Axis 1 is the columns in 2d array
            self.cumsum_mine = np.cumsum(underground, axis=1)
        else:
            # 3d Mine
            self.len_y, self.len_x, self.len_z = underground.shape
            # Initial state 0's as nothing has been mined
            self.initial = convert_to_tuple(np.zeros(( self.len_y, self.len_x), int))
            # Axis 2 is the columns in 3d array
            self.cumsum_mine = np.cumsum(underground, axis=2)
        # Print mine X, Y, Z and mine shape to console for sanity check
        # print(f'X: {self.len_x}  Y:{self.len_y}  Z:{self.len_z}')
        # print(f'CUMSUM of columns: {self.cumsum_mine}')
        # self.console_display()
        # Display graph of initial mind to double check for sanity sake
        # self.plot_state(self.initial)

    def surface_neigbhours(self, loc):
        '''
        Return the list of neighbours of loc
        Parameters
        ----------
        loc : surface coordinates of a cell.
            a singleton (x,) in case of a 2D mine
            a pair (x,y) in case of a 3D mine
        Returns
        -------
        A list of tuples representing the surface coordinates of the
        neighbouring surface cells.
        '''
        L = []
        assert len(loc) in (1, 2)
        if len(loc) == 1:
            if loc[0]-1 >= 0:
                L.append((loc[0]-1,))
            if loc[0]+1 < self.len_x:
                L.append((loc[0]+1,))
        else:
            # len(loc) == 2
            for dx, dy in ((-1, -1), (-1, 0), (-1, +1),
                           (0, -1), (0, +1),
                           (+1, -1), (+1, 0), (+1, +1)):
                if (0 <= loc[0]+dx < self.len_x) and (0 <= loc[1]+dy < self.len_y):
                    L.append((loc[0]+dx, loc[1]+dy))
        return L

    def actions(self, state):
        '''
        Return a generator of valid actions in the give state 'state'
        An action is represented as a location. An action is deemed valid if
        it doesn't  break the dig_tolerance constraint.
        Parameters
        ----------
        state : 
            represented with nested lists, tuples or a ndarray
            state of the partially dug mine
        Returns
        -------
        a generator of valid actions
        '''
        # Convert state dict to np array
        state = np.array(state)
        # Make sure the state is either 2d or 1d
        assert state.ndim == 1 or state.ndim == 2

        # Create a generator of valid actions
        def generator_valid_actions(): 
            '''
            Enumerate through each position of the mine and increment an action.  If
            the action is valid it is returned by this generator function.  The position
            is returned for instance 1D state: (x) and 2D state: (x,y)
            '''
            for i, col in enumerate(state):
                # 1d state - create copy of state and increment digging one cell
                if state.ndim == 1:
                    new_state = np.array(state)
                    new_state[i] += 1
                    # If the new state is not dangerous yield new state in generator
                    if(not self.is_dangerous(new_state) and col < self.len_z): # Don't break z bounds
                        yield (i,)
                else:
                    # 2d state - create copy and increment digging one cell
                    for y, cell in enumerate(col):
                        new_state = np.array(state)
                        new_state[i, y] += 1
                        # If the new state is not dangerous yield new state in generator
                        if(not self.is_dangerous(new_state) and cell < self.len_z):
                            yield tuple((i, y))
        # Return Generator Function
        return generator_valid_actions()

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must a valid actions.
        That is, one of those generated by  self.actions(state)."""
        action = tuple(action)
        new_state = np.array(state)  # Make a copy
        new_state[action] += 1
        return convert_to_tuple(new_state)

    def console_display(self):
        '''
        Display the mine on the console
        Returns
        -------
        None.
        '''
        print('Mine of depth {}'.format(self.len_z))
        if self.underground.ndim == 2:
            # 2D mine
            print('Plane x,z view')
        else:
            # 3D mine
            print('Level by level x,y slices')
        #
        print(self.__str__())

    def __str__(self):
        if self.underground.ndim == 2:
            # 2D mine
            return str(self.underground.T)
        else:
            # 3D mine
            # level by level representation
            return '\n'.join('level {}\n'.format(z)
                             + str(self.underground[..., z]) for z in range(self.len_z))

            return self.underground[loc[0], loc[1], :]

    @staticmethod
    def plot_state(state):
        if state.ndim == 1:
            fig, ax = plt.subplots()
            ax.bar(np.arange(state.shape[0]),
                   state
                   )
            ax.set_xlabel('x')
            ax.set_ylabel('z')
        else:
            assert state.ndim == 2
            # bar3d(x, y, z, dx, dy, dz,
            # fake data
            _x = np.arange(state.shape[0])
            _y = np.arange(state.shape[1])
            _yy, _xx = np.meshgrid(_y, _x)  # cols, rows
            x, y = _xx.ravel(), _yy.ravel()
            top = state.ravel()
            bottom = np.zeros_like(top)
            width = depth = 1
            fig = plt.figure(figsize=(3, 3))
            ax1 = fig.add_subplot(111, projection='3d')
            ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_zlabel('z')
            ax1.set_title('State')
        #
        plt.show()

    def payoff(self, state):
        '''
        Compute and return the payoff for the given state.
        That is, the sum of the values of all the digged cells.
        No loops needed in the implementation!        
        '''
        cumsum = np.array(self.cumsum_mine)
        state = np.array(state)

        assert cumsum.ndim == 2 or cumsum.ndim == 3
        assert state.ndim == 1 or state.ndim == 2

        # If the state is from 3d min and cumsum is from 2d mine, prefer the cumsum
        # flatten the state so its workable by the 2d method
        if state.ndim == 2 and cumsum.ndim == 2:
            state = state.flatten()

        if cumsum.ndim == 2:
            # 2d Mine
            rows = np.where(state != 0)
            i = state[rows] - 1
            q = cumsum[rows, i]
        else :
            # 3d Mine
            x, y = np.where(state != 0)
            i = state[x, y] - 1
            q = cumsum[x, y, i]
        return np.sum(q, dtype='float32')

    def is_dangerous(self, state):
        '''
        Return True iff the given state breaches the dig_tolerance constraints.
        No loops needed in the implementation!
        '''
        # convert to np.array in order to use numpy operators
        state = np.array(state)
        # Make sure the state array complies with a 1d or 2d state
        assert state.ndim == 1 or state.ndim == 2

        # Broken into functions incase of re use
        def create_breach_map_diagonals(state):
            '''
            Create a breach map on the diagonal with a breach defined as
            | state[col] - state[col2] |
            '''
            return np.abs(state[:-1, :-1] - state[1:, 1:])
        def create_breach_map_vertical(state):
            '''
            Create a breach map on the vertical axis with a breach defined as
            | state[col] - state[col2] |
            '''
            return np.abs(state[:, 1:] - state[:, :-1])
        def create_breach_map_horizontal(state):
            '''
            Create a breach map on the horizontal axis with a breach defined as
            | state[col] - state[col2] |
            '''
            return np.abs(state[1:, :] - state[:-1, :])
        def create_breach_map_1d_state(state):
            '''
            Create a breach map for a 1d array state.  Breach is defined as
            | state[col] - state[col2] |
            '''
            return np.abs(state[:-1] - state[1:])
        def check_for_breaches(breach_map):
            '''
            Check a created breach map for any vales that are not within the dig tolerance range.
            If there was any breaches return true for breach found.
            '''
            tolerance_breaches = np.where(breach_map > self.dig_tolerance, True, False)
            return np.any(tolerance_breaches == True)

        # If the mine is 2d the state is 1d, check the neighbors in 1d state
        if state.ndim == 1:
            if(check_for_breaches(create_breach_map_1d_state(state))): 
                return True
            else:
                # Return false if no breaches were found
                return False

        # If mine is 3d the state is 2d, check the neighbors in 2d state for any breaches
        # Return immediately to speed up method
        if(check_for_breaches(create_breach_map_vertical(state))): 
            return True
        if(check_for_breaches(create_breach_map_horizontal(state))): 
            return True
        if(check_for_breaches(create_breach_map_diagonals(state))): 
            return True
        # Flip the array so the other diagonals can be checked
        state = np.flip(state, axis=1)
        if(check_for_breaches(create_breach_map_diagonals(state))): 
            return True
        
        # Fall through --> if no breaches were found return false as the state is not dangerous
        return False
    # ========================  Class Mine  ==================================


def search_dp_dig_plan(mine):
    '''
    Search using Dynamic Programming the most profitable sequence of 
    digging actions from the initial state of the mine.
    Return the sequence of actions, the final state and the payoff
    Parameters
    ----------
    mine : a Mine instance
    Returns
    -------
    A recursive function includes best_payoff, best_action_list, best_final_stage
    '''   
    # Use functools.lru_cache as memoization
    @functools.lru_cache(maxsize = 2**20)
    def search_rec(s):
        '''
        Using recursive for the DP method
        Parameters
        ----------
        s : state of the mine
        Returns
        -------
        best_payoff, best_action_list, best_final_stage
        '''
        best_payoff = mine.payoff(s)
        best_action_list = []
        best_final_stage = s
        
        # Breaking states down to child states as temporary actions 
        for temp_action in mine.actions(s):
            # Get the child state via mine.results
            child_state = mine.result(s, temp_action) 
            temp_payoff, temp_action_list, temp_final_stage = search_rec(child_state)

            # Check if temp_payoff is better than best_payoff
            if best_payoff < temp_payoff:
                best_payoff = temp_payoff
                best_action_list = [temp_action] +  list(temp_action_list)
                best_final_stage = temp_final_stage
        
        # Convert to tuple to fit the assignment's requirement
        best_action_list = convert_to_tuple(best_action_list)
        best_final_stage = convert_to_tuple(best_final_stage)
        return best_payoff, best_action_list, best_final_stage
    
    return search_rec(mine.initial)
    #levm = lru_cache(maxsize=1024)(search_rec)

def optimise_cols(underground, state):
    underground = np.array(underground)
    state = np.array(state)
    # Make sure this is correct dimensions
    assert (underground.ndim == 3 and state.ndim == 2) or underground.ndim == 2 and state.ndim == 1

    if underground.ndim == 3 and state.ndim == 2:
        # 3d Mine
        col_range = np.arange(underground.shape[2])
        # Translate and take 1 for indexing
        state_trans = state[:, :, np.newaxis] - 1
        # Set axis operator for final cumsum max
        axis_operator = 2
    else:
        #2d Mine
        col_range = np.arange(underground.shape[1])
        # Translate and take 1 for indexing
        state_trans = np.reshape(state, (-1, 1)) - 1
        # Set axis operator for final cumsum max
        axis_operator = 1
    # Create a state map, where dug cells are true
    underground_dug_state_map = (state_trans >= col_range) & (state_trans > -1)
    #From state map set dug cells to 0 --> only interested in the best values 
    #No including dug cells
    underground[:len(underground_dug_state_map)][underground_dug_state_map] = 0
    # REturn a cum sum max of the optimised values
    return np.amax(np.cumsum(underground, axis=axis_operator), axis=axis_operator)

def search_bb_dig_plan(mine):
    '''
    Compute, using Branch and Bound, the most profitable sequence of 
    digging actions from the initial state of the mine.
    Parameters
    ----------
    mine : Mine
        An instance of a Mine problem.
    Returns
    -------
    best_payoff, best_action_list, best_final_state
    '''
    frontier = []
    # Variables to store the best state and best state cost
    best_state = mine.initial
    best_payoff = mine.payoff(best_state)
    # Add the initial state to the tree
    frontier.append(mine.initial)
    while frontier:
        # Pop new state from frontier
        state = frontier.pop()
        # Calculate b(s)
        b = optimise_cols(mine.underground, state)
        # Prune if there isnt a sub tree with a better payoff
        if b.max() + mine.payoff(state) <= best_payoff:
            continue
        # Explore child states based on valid action
        for action in mine.actions(state):
            # Calculate the result from the action
            result = mine.result(state, action)
            # Calculate the payoff from the result
            pay_off = mine.payoff(result)

            # If the pay off is better than the current pay off, save that state
            if  pay_off > best_payoff:
                best_state = result
                best_payoff = pay_off
            # Append child state to frointer
            frontier.append(result)
    return best_payoff, find_action_sequence(mine.initial, best_state), best_state


def find_action_sequence(s0, s1):
    '''
    Compute a sequence of actions to go from state s0 to state s1.
    There may be several possible sequences.
    Preconditions: 
        s0 and s1 are legal states, s0<=s1 and 
    Parameters
    ----------
    s0 : tuple based mine state
    s1 : tuple based mine state 
    Returns
    -------
    A sequence of actions to go from state s0 to state s1
    '''
    action_sequence = []
    # convert to no arrays
    s0 = np.array(s0)
    s1 = np.array(s1)
    
    # Ensure the states passed to the function are of a valid size
    assert s0.ndim == 1 or s0.ndim == 2
    assert s1.ndim == 1 or s1.ndim == 2
    assert s0.ndim == s1.ndim

    # Keep looping through until mine state matches
    while not np.array_equal(s0, s1):
        for x in range(s0.shape[0]):
            # For a 2d mine state
            if s0.ndim == 1:
                # If the index has reached disired index, pune
                if s0[x] == s1[x]:
                    continue
                # From dug state to initial
                if s0[x] >= s1[x]:
                    action_sequence.append((x, ))
                    s0[x] -= 1
                # From initial state to dug state
                else:
                    action_sequence.append((x, ))
                    s0[x] += 1
            else: 
                # 3d mine state
                for y in range(s0.shape[1]):
                    # If the index has reached disired index, pune
                    if s0[x, y] == s1[x, y]:
                        continue
                    # From dug state to initial
                    if s0[x, y] >= s1[x, y]:
                        action_sequence.append((x, y))
                        s0[x, y] -= 1
                    # From initial state to dug state
                    else:
                        action_sequence.append((x, y))
                        s0[x, y] += 1
    return action_sequence
