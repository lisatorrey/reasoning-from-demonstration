"""Tools for imitation learning in Taxi."""

import gym

from random import random, choice

from rfd.qfunction import QFunction

# Taxi environment
ENV = gym.make('Taxi-v2')
ACTIONS = [0, 1, 2, 3, 4, 5]

# Learning parameters
ALPHA = 0.1
GAMMA = 0.9
EPSILON_MAX = 0.1
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.99

# Curve configurations
EPISODES = 4000
WINDOW = 400
FREQ = 40


class Driver(object):
    """Agent for learning Taxi, possibly using imitation and/or decomposition."""
    
    def __init__(self):
        self.policy = QFunction(ALPHA, GAMMA)
        self.epsilon = EPSILON_MAX
        self.curve = list()

    def exploit(self, s):
        """Return an action with the maximal Q-value in the given state."""
        qvalues = {a: self.policy.Q(s, a) for a in ACTIONS}
        best = max(qvalues.values())
        return choice([a for a, q in qvalues.items() if q >= best])

    def generate(self, demos):
        """Add an episode of demonstrated state-action pairs to the given mapping."""
        done = False
        obs = ENV.reset()
        
        # Remember states both ways
        s = Driver.state(obs, decompose=False)
        sd = Driver.state(obs, decompose=True)

        # Save the exploit action
        while not done:
            action = self.exploit(s)
            demos[s] = action
            demos[sd] = action

            # Continue without learning
            obs, reward, done, info = ENV.step(action)
            s = Driver.state(obs, decompose=False)
            sd = Driver.state(obs, decompose=True)

    def train(self, demos=None, decompose=False):
        """Produce a learning curve."""
        length = 0
        scores = list()

        # Begin an episode
        for episode in range(EPISODES):
            score = 0
            done = False
            obs = ENV.reset()
            s = Driver.state(obs, decompose)

            # Choose an action
            while not done:
                if random() < self.epsilon:
                    action = choice(ACTIONS)
                elif demos is not None and s in demos:
                    action = demos[s]
                else:
                    action = self.exploit(s)
                
                # Perform the action
                obs, reward, done, info = ENV.step(action)
                sp = Driver.state(obs, decompose)
                score += reward
                length += 1
                
                # Learn from it
                if done:
                    self.policy.update(s, action, reward)
                    self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
                else:
                    self.policy.update(s, action, reward, sp, ACTIONS)
                    s = sp

            # Add to the curve
            scores.append(score)
            if episode % FREQ == 0:
                average = sum(scores[-WINDOW:]) / len(scores[-WINDOW:])
                self.curve.append((length, average))

    @staticmethod
    def state(obs, decompose=False):
        """Return a tuple of features for the given observation, possibly decomposing into subtasks."""
        row, column, passenger, destination = tuple(ENV.decode(obs))
        if not decompose:
            return row, column, passenger, destination  # Global state
        elif passenger < 4:
            return "pickup", row, column, passenger  # Subtask 1 in an optimal decomposition
        else:
            return "dropoff", row, column, destination  # Subtask 2 in an optimal decomposition


class Comparison(object):
    """Collection of agents trained with varying levels of imitation."""
    
    def __init__(self, levels):
        self.independent = Driver()  # Generates demos for the others
        self.imitators = {level: Driver() for level in levels}
        self.decomposers = {level: Driver() for level in levels}

        self.independent.train()
        demos = dict()
        num_demos = 0

        for level in levels:
            while num_demos < level:
                self.independent.generate(demos)
                num_demos += 1

            self.imitators[level].train(demos)
            self.decomposers[level].train(demos, decompose=True)


class Plot(object):
    """Collection of learning curves."""
    
    def __init__(self):
        self.curves = list()
    
    def add(self, drivers):
        """Add a curve by averaging the curves of the given agents."""
        clusters = zip(*(driver.curve for driver in drivers))
        average = lambda cluster, i: sum(point[i] for point in cluster) / len(cluster)
        curve = [(average(cluster, 0), average(cluster, 1)) for cluster in clusters]
        self.curves.append(curve)
    
    def save(self, filename):
        """Write all the curves to a file."""
        f = open(filename, "w")
        indent = ""
        for curve in self.curves:
            indent += "\t"
            for x, y in curve:
                f.write(str(x) + indent + str(y) + "\n")
        f.close()
