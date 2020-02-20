"""Tools for reinforcement learning."""


class QFunction(object):
    """Map-based Q-function."""

    def __init__(self, alpha=1.0, gamma=1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.q = dict()

    def Q(self, s, a):
        """Return a Q-value estimate for the given step."""
        return self.q[s][a] if s in self.q and a in self.q[s] else 0

    def delta(self, s, a, r, sp=None, actions=None):
        """Return a Q-value change produced by the given observation."""
        delta = r - self.Q(s, a)
        if sp is not None and actions is not None and len(actions) > 0:
            delta += self.gamma * max(self.Q(sp, ap) for ap in actions)
        return delta

    def update(self, s, a, r, sp=None, actions=None):
        """Update a Q-value based on the given observation."""
        delta = self.delta(s, a, r, sp, actions)
        if s not in self.q:
            self.q[s] = dict()
        if a not in self.q[s]:
            self.q[s][a] = 0
        self.q[s][a] += self.alpha * delta
