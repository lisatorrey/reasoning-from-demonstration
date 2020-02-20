"""The RfD agent."""

from math import exp
from pickle import dump
from random import choice, random

from rfd.search import Map
from rfd.theory import Theory
from rfd.qfunction import QFunction

ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor

BONUS = 100  # For completing an objective
LIMIT = 10000  # Maximum attempt length

BETA_MAX = 100  # Highest risk weight
BETA_DECAY = 0.99  # Decrease in risk weight

EPSILON_MAX = 0.1  # Highest exploration rate
EPSILON_MIN = 0.01  # Lowest exploration rate
EPSILON_DECAY = 0.99  # Decrease in exploration rate


class Agent(object):
    def __init__(self, extend_theory=True, extend_map=True):
        self.extend_theory = extend_theory
        self.extend_map = extend_map

        # Knowledge base
        self.theory = Theory()
        self.map = Map()

        # Policy collection
        self.routes = dict()
        self.tactics = dict()
        self.reflexes = dict()

    def observe(self, record):
        """Watch a demonstration and construct knowledge accordingly."""
        for frame in record:
            self.theory.update(frame)
            self.map.update(frame)

    def save(self, filename):
        """Save this agent for later use."""
        self.prepare()
        try:
            f = open(filename, "wb")
        except FileNotFoundError:
            filename = input("Need a different agent filename: ")
            self.save(filename)
            return
        try:
            dump(self, f)
        except MemoryError:
            print("MemoryError while trying to save", filename)
        f.close()

    def prepare(self):
        """Reset short-term data before attempting a task."""
        self.action = None
        self.objective = None
        self.checkpoint = None
        self.antiobjectives = set()

        # Reset risk weights
        for policy in self.routes.values():
            policy.beta = BETA_MAX
        for policy in self.tactics.values():
            policy.beta = BETA_MAX

    def attempt(self, task):
        """Try to complete the given task."""
        self.prepare()
        while not task.ended():
            task.perform(self.act(task))
            task.update()
            self.update(task)
            if len(task.record) > LIMIT:
                break

    def act(self, task, verbose=False):
        """Choose an action in the given task."""
        actions = task.actions()

        # Choose an objective
        previous_objective = self.objective
        previous_checkpoint = self.checkpoint
        self.strategize(task)

        # Report intentions
        if verbose and (self.objective, self.checkpoint) != (previous_objective, previous_checkpoint):
            if self.checkpoint is None:
                print("Objective:", str(self.objective))
            else:
                print("Objective:", str(self.objective), "via", self.checkpoint.subject.region)

        # Deploy a policy
        if self.checkpoint is not None:
            policy = self.routes[self.checkpoint.template]
            rewards = {a: policy.Q(self.checkpoint.s, a) for a in actions}
            epsilon = policy.epsilon
        elif self.objective is not None:
            policy = self.tactics[self.objective.template]
            rewards = {a: policy.Q(self.objective.s, a) for a in actions}
            epsilon = policy.epsilon
        else:
            epsilon = 1.0

        # Evaluate risks
        risks = {a: 0 for a in actions}
        for a in actions:
            for objective in self.antiobjectives:
                risks[a] -= self.reflexes[objective.template].Q(objective.s, a)

        # Choose an action
        if random() < epsilon:
            safest = min(risks.values())
            self.action = choice([a for a in actions if risks[a] <= safest])
        else:
            values = {a: rewards[a] - policy.beta * risks[a] for a in actions}
            best = max(values[a] for a in actions)
            self.action = choice([a for a in actions if values[a] >= best])
            policy.beta = policy.beta * BETA_DECAY

        return self.action

    def strategize(self, task):
        """Choose an objective in the given task."""
        self.checkpoint = None
        self.objective = None

        # Identify undesirable events
        self.antiobjectives = set()
        for template in self.theory.causes("FAILURE"):
            if template not in self.reflexes:
                self.reflexes[template] = QFunction(ALPHA)
            for objective in task.frame.objectives(template):
                self.antiobjectives.add(objective)
                objective.s = objective.state()

        # Identify desirable events
        objectives = set()
        if not task.ended():
            for template in self.theory.contributors("SUCCESS", task.frame):
                objectives |= task.frame.objectives(template)

        # Choose the nearest objective
        if len(objectives) > 0:
            sources = {objective.actor for objective in objectives}
            targets = {objective.subject for objective in objectives}
            searches = {source: self.map.search(source, targets) for source in sources}
            distances = {objective: searches[objective.actor].distance(objective.subject) for objective in objectives}
            self.objective = choice([objective for objective, d in distances.items() if d == min(distances.values())])

            # Prepare the objective
            self.objective.s = self.objective.state()
            self.objective.d = self.objective.distance()
            if self.objective.template not in self.tactics:
                self.tactics[self.objective.template] = QFunction(ALPHA, GAMMA)
                self.tactics[self.objective.template].epsilon = EPSILON_MAX
                self.tactics[self.objective.template].beta = BETA_MAX

            # Prepare a checkpoint
            if not self.objective.regional():
                if not searches[self.objective.actor].found(self.objective.subject):
                    self.objective = None
                else:
                    self.checkpoint = searches[self.objective.actor].checkpoint(self.objective)
                    self.checkpoint.s = self.checkpoint.state()
                    self.checkpoint.d = self.checkpoint.distance()
                    if self.checkpoint.template not in self.routes:
                        self.routes[self.checkpoint.template] = QFunction(ALPHA, GAMMA)
                        self.routes[self.checkpoint.template].epsilon = EPSILON_MAX
                        self.routes[self.checkpoint.template].beta = BETA_MAX

    def update(self, task):
        """Reflect on the step just taken."""

        # Update knowledge
        if self.extend_theory:
            self.theory.update(task.frame)
        if self.extend_map:
            self.map.update(task.frame)

        # Learn about routes
        if self.checkpoint is not None:
            policy = self.routes[self.checkpoint.template]
            progress = self.checkpoint.d - self.checkpoint.distance()
            if self.checkpoint.regional():
                policy.update(self.checkpoint.s, self.action, progress + BONUS)
                policy.epsilon = max(EPSILON_MIN, policy.epsilon * EPSILON_DECAY)
                policy.beta = BETA_MAX
            elif task.frame.transitions(self.checkpoint.actor):
                policy.update(self.checkpoint.s, self.action, progress - BONUS)
            else:
                policy.update(self.checkpoint.s, self.action, progress, self.checkpoint.state(), task.actions())

        # Learn about objectives
        elif self.objective is not None:
            policy = self.tactics[self.objective.template]
            progress = self.objective.d - self.objective.distance()
            if self.objective in task.frame.events:
                policy.update(self.objective.s, self.action, progress + BONUS)
                policy.epsilon = max(EPSILON_MIN, policy.epsilon * EPSILON_DECAY)
                policy.beta = BETA_MAX
            elif task.frame.supports(self.objective):
                policy.update(self.objective.s, self.action, progress, self.objective.state(), task.actions())

        # Learn about risks
        for objective in self.antiobjectives:
            policy = self.reflexes[objective.template]
            if objective in task.frame.events:
                policy.update(objective.s, self.action, -BONUS)
            elif task.frame.supports(objective):
                policy.update(objective.s, self.action, 0, objective.state(), task.actions())
