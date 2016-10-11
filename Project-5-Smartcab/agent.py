import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
import pandas as pd

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        global Q_dict
        Q_dict ={}
        from collections import defaultdict
        Q_dict= defaultdict(lambda:0, Q_dict)

        self.epsilon = 0.1
        self.counter = 1.0
        self.alpha = 1
        self.k= 5 
        self.successes = []
        self.penalty = 0
        self.total_steps = 0
        
        
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.successes.append(0)
        self.action = random.choice([None, 'forward', 'left', 'right'])
        self.state = random.choice([None, 'forward', 'left', 'right'])
        self.next_state = random.choice([None, 'forward', 'left', 'right'])
        
        
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = (self.next_waypoint, inputs['light'], inputs['oncoming'], inputs['left'])
        
        # TODO: Select action according to your policy
        
        def choose_action(self, state):
            actions = [None, 'forward', 'left', 'right']
            q = [Q_dict[(state, a)] for a in actions]
            maxQ = max(q)
            count = q.count(maxQ)
            if count >1:
                same_values = [i for i in range(len(actions)) if q[i] == maxQ]
                action_choices = [actions[same_values] for same_values in range(len(same_values))] 
                best_action = random.choice(action_choices)
            elif random.random() < self.epsilon:
                best_action = random.choice(Environment.valid_actions)
            else:
                i = q.index(maxQ)
                best_action = actions[i]
            return best_action

        
        self.epsilon = 1.0/(self.counter)
        self.counter += 0.01
        print self.epsilon

        action = choose_action(self,self.state)

       
        # Execute action and get reward
        reward = self.env.act(self, action)

        
        # TODO: Learn policy based on state, action, reward
        #alpha = 0.2
        self.k += 0.5
        gamma = 0.1
        
        self.new_next_waypoint= self.planner.next_waypoint()
        inputs_new = self.env.sense(self)
        self.next_state = (self.new_next_waypoint, inputs_new['light'], inputs_new['oncoming'], inputs_new['left'])
        next_action = choose_action(self, self.next_state)

        Q_dict[(self.state,action)]=(1-self.alpha)*Q_dict[(self.state, action)] + self.alpha*(reward+gamma*Q_dict[(self.next_state, next_action)])
        
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

        location = self.env.agent_states[self]["location"] 
        destination = self.env.agent_states[self]["destination"]

        if location == destination:
            self.successes[-1] = 1

        success_rate = (self.successes.count(1))/float(100)
        print("success rate: " "%.2f" % success_rate)
        
        self.total_steps+=1
        
        if reward < 0: #Assign penalty if reward is negative
            self.penalty += 1
        print("number of penalties: " "%i" % self.penalty)
        print("percentage of penalties: " "%.2f" % (self.penalty/(float(self.total_steps))))

        print "inputs = {}".format(inputs)
        print "next_waypoint: %s" % self.next_waypoint
        print "action: %s" % action
        
def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.005, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
