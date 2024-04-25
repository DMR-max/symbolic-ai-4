#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Programming 
Practical for course 'Symbolic AI'
2020, Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from world import World

class Dynamic_Programming:

    def __init__(self):
        self.V_s = None # will store a potential value solution table
        self.Q_sa = None # will store a potential action-value solution table
        
    def value_iteration(self,env,gamma = 1.0, theta=0.001):
        ''' Executes value iteration on env. 
        gamma is the discount factor of the MDP
        theta is the acceptance threshold for convergence '''

        print("Starting Value Iteration (VI)")

        # initialize value table
        V_s = np.zeros(env.n_states)

        max_val = 0

        # Start value iteration
        while True:
            max_diff = 0  
            V_s_copy = np.zeros(env.n_states)

            for s in env.states:
                max_val = 0

                for a in env.actions:
                    # Compute state value
                    state, val  = env.transition_function(s,a)
                    val +=  gamma * V_s[state]

                    # Store value best action so far
                    max_val = max(max_val, val)

                    # Update value with highest value
                    V_s_copy[s] = max_val

                # Update maximum difference
                max_diff = max(max_diff, abs(V_s[s] - V_s_copy[s]))

            # Update value functions
            V_s = V_s_copy

            # If diff smaller than threshold theta for all states, breaks from while loop
            if max_diff < theta:
                break

        self.V_s = V_s
        print(V_s)

        return

    def Q_value_iteration(self,env,gamma = 1.0, theta=0.001):
        ''' Executes Q-value iteration on env. 
        gamma is the discount factor of the MDP
        theta is the acceptance threshold for convergence '''

        print("Starting Q-value Iteration (QI)")

        # initialize state-action value table
        Q_sa = np.zeros([env.n_states,env.n_actions])

        max_val = 0

        # Start value iteration
        while True:
            max_diff = 0
            Q_sa_copy = np.zeros([env.n_states,env.n_actions])

            for s in env.states:
                max_val = 0

                for a in env.actions:

                    for p in env.actions:
                        # Compute state value
                        state, val = env.transition_function(s,a) 
                        val +=  gamma * Q_sa[state][np.where(env.actions == p)]

                        # Store value best action so far
                        max_val = max(max_val, val)

                    # Update value with highest value
                    Q_sa_copy[s][np.where(env.actions == a)] = max_val 

                    # Update maximum difference
                    max_diff = max(max_diff, abs(Q_sa[s][np.where(env.actions == a)] - Q_sa_copy[s][np.where(env.actions == a)]))

            # Update value functions
            Q_sa = Q_sa_copy

            # If diff smaller than threshold theta for all states, breaks from while loop
            if max_diff < theta:
                break       

        self.Q_sa = Q_sa
        print(Q_sa)

        return
                
    def execute_policy(self,env,table='V'):
        ## Execute the greedy action, starting from the initial state
        env.reset_agent()
        print("Start executing. Current map:") 
        env.print_map()
        while not env.terminal:
            current_state = env.get_current_state() # this is the current state of the environment, from which you will act
            available_actions = env.actions
            # Compute action values
            if table == 'V' and self.V_s is not None:

                env.print_state(current_state)
                max = 0

                # Loop trough all actions
                for a in env.actions:
                    state, rew = env.transition_function(current_state,a)
                    val = rew + self.V_s[state]

                    # Check if value is bigger then max value
                    if val > max:
                        max = val
                        action = a

                greedy_action = action

                
            
            elif table == 'Q' and self.Q_sa is not None:

                env.print_state(current_state)

                # Get index of action with highest value
                best = np.argmax(self.Q_sa[current_state])

                # Check what action the index is
                action = available_actions[best]

                greedy_action = action
                
                
            else:
                print("No optimal value table was detected. Only manual execution possible.")
                greedy_action = None


            # ask the user what he/she wants
            while True:
                if greedy_action is not None:
                    print('Greedy action= {}'.format(greedy_action))    
                    your_choice = input('Choose an action by typing it in full, then hit enter. Just hit enter to execute the greedy action:')
                else:
                    your_choice = input('Choose an action by typing it in full, then hit enter. Available are {}'.format(env.actions))
                    
                if your_choice == "" and greedy_action is not None:
                    executed_action = greedy_action
                    env.act(executed_action)
                    break
                else:
                    try:
                        executed_action = your_choice
                        env.act(executed_action)
                        break
                    except:
                        print('{} is not a valid action. Available actions are {}. Try again'.format(your_choice,env.actions))
            print("Executed action: {}".format(executed_action))
            print("--------------------------------------\nNew map:")
            env.print_map()
        print("Found the goal! Exiting \n ...................................................................... ")
    

def get_greedy_index(action_values):
    ''' Own variant of np.argmax, since np.argmax only returns the first occurence of the max. 
    Optional to uses '''
    return np.where(action_values == np.max(action_values))
    
if __name__ == '__main__':
    env = World('prison.txt') 
    DP = Dynamic_Programming()

    # Run value iteration
    input('Press enter to run value iteration')
    optimal_V_s = DP.value_iteration(env)
    input('Press enter to start execution of optimal policy according to V')
    DP.execute_policy(env, table='V') # execute the optimal policy
    
    # Once again with Q-values:
    input('Press enter to run Q-value iteration')
    optimal_Q_sa = DP.Q_value_iteration(env)
    input('Press enter to start execution of optimal policy according to Q')
    DP.execute_policy(env, table='Q') # execute the optimal policy

