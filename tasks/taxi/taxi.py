"""Task defined in Taxi from OpenAI Gym."""

import gym

from time import sleep

from rfd.task import TaskInterface
from rfd.event import Object, Event

DELAY = 1  # Seconds between renders


class Task(TaskInterface):
    def __init__(self):
        self.env = gym.make("Taxi-v2")
        self.current_events = set()
        self.total_reward = 0
        self.success = False
        self.done = False

        # Objects
        r, c, p, d = self.env.decode(self.env.reset())
        self.taxi = Object("Taxi", (r, c), Task.region((r, c)))
        self.passenger = Object("Passenger", Task.locate(p), Task.region(p))
        self.destination = Object("Destination", Task.locate(d), Task.region(d))
        self.stops = {Object("Stop", Task.locate(s), Task.region(s)) for s in {0, 1, 2, 3} - {p, d}}

        TaskInterface.__init__(self)

    def actions(self):
        """Return a set of action choices."""
        return {0, 1, 2, 3, 4, 5}

    def perform(self, action):
        """Perform the given action."""
        self.current_events = set()

        # Step
        state, reward, done, info = self.env.step(action)
        r, c, p, d = self.env.decode(state)
        self.total_reward += reward
        self.success = (p == d)
        self.done = done

        # Update objects
        self.taxi.location = (r, c)
        self.taxi.region = Task.region((r, c))

        # Detect success
        if self.success:
            self.current_events.add(Event("drops", self.taxi, self.destination))
            self.stops.add(Object("Stop", self.taxi.location, self.taxi.region))
            self.destination = None

        # Detect pickup
        elif p == 4 and self.taxi.type == "Taxi":
            self.current_events.add(Event("picks", self.taxi, self.passenger))
            self.taxi = Object("Taxi+Passenger", self.taxi.location, self.taxi.region)
            self.stops.add(Object("Stop", self.taxi.location, self.taxi.region))
            self.passenger = None

        # Detect abandonment
        elif p < 4 and self.taxi.type == "Taxi+Passenger":
            stop = {stop for stop in self.stops if stop.location == self.taxi.location}.pop()
            self.current_events.add(Event("drops", self.taxi, stop))
            self.taxi = Object("Taxi", self.taxi.location, self.taxi.region)
            self.passenger = Object("Passenger", self.taxi.location, self.taxi.region)
            self.stops.remove(stop)

    def objects(self):
        """Return a set of perceived objects."""
        return self.stops | {obj for obj in {self.taxi, self.passenger, self.destination} if obj is not None}

    def events(self):
        """Return a set of perceived events."""
        return self.current_events

    def succeeded(self):
        """Return whether this attempt has succeeded."""
        return self.done and self.success

    def failed(self):
        """Return whether this attempt has failed."""
        return self.done and not self.success

    def ended(self):
        """Return whether this attempt has ended."""
        return self.done

    def score(self):
        """Return a score for this attempt."""
        return self.total_reward

    def length(self):
        """Return a length for this attempt."""
        return len(self.record)

    def demonstrate(self): 
        """Allow a person to demonstrate this task."""
        self.env.render()
        while not self.ended():
            action = Task.act(input("Action: "))
            self.perform(action)
            self.update()
            self.env.render()

    def display(self, agent):
        """Show the given agent attempting this task."""
        agent.prepare()
        self.env.render()
        while not self.ended():
            self.perform(agent.act(self, verbose=True))
            self.update()
            agent.update(self)
            sleep(DELAY)
            self.env.render()

    @staticmethod
    def act(symbol):
        """Return an action for the given key."""
        if symbol == "s":
            return 0  # south
        elif symbol == "w":
            return 1  # north
        elif symbol == "d":
            return 2  # east
        elif symbol == "a":
            return 3  # west
        elif symbol == "p":
            return 4  # pickup
        elif symbol == "o":
            return 5  # dropoff

    @staticmethod
    def locate(stop):
        """Return the row,col location of the given stop."""
        if stop == 0:
            return 0, 0  # R
        elif stop == 1:
            return 0, 4  # G
        elif stop == 2:
            return 4, 0  # Y
        else:  # 3
            return 4, 3  # B

    @staticmethod
    def region(location_or_stop):
        """Return a region label for the given location or stop."""
        r, c = location_or_stop if isinstance(location_or_stop, tuple) else Task.locate(location_or_stop)
        if r < 3:
            return "R" if c < 2 else "G"
        else:
            return "Y" if c < 1 else "middle" if c < 3 else "B"
