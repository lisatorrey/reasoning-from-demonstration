"""Task defined in Montezuma's Revenge from OpenAI Gym."""

import gym
import pyglet
import numpy as np

from time import sleep
from random import randrange

from rfd.task import TaskInterface
from rfd.event import Object, Event

SLOW = 0.03  # Seconds between renders for playability
FAST = 0.01  # Seconds between renders for watchability


class Joe(Object):
    def __init__(self):
        self.dead = False
        Object.__init__(self, "Joe", (79, 81), region=Task.region((79, 81)))

    def update(self, image):
        """Find Joe based on his color."""
        indices = np.where(image[50:, :, 0] == 200)
        if len(indices[0]) > 0:
            x = (np.amin(indices[1]) + np.amax(indices[1])) // 2
            y = 50 + (np.amin(indices[0]) + np.amax(indices[0])) // 2
            self.velocity = (x - self.location[0], y - self.location[1])
            self.region = Task.region((x, y))
            self.size = len(indices[0])
            self.location = (x, y)


class Skull(Object):
    def __init__(self):
        self.dead = False
        Object.__init__(self, "Skull", (93, 172), region=Task.region((93, 172)))

    def update(self, image):
        """Find the skull based on its color."""
        indices = np.where(image[50:, :, 0] == 236)
        if len(indices[0]) > 0:
            x = (np.amin(indices[1]) + np.amax(indices[1])) // 2
            y = 50 + (np.amin(indices[0]) + np.amax(indices[0])) // 2
            self.velocity = (x - self.location[0], y - self.location[1])
            self.location = (x, y)


class Task(TaskInterface):
    def __init__(self):
        self.env = gym.make("MontezumaRevengeNoFrameskip-v4")
        self.current_events = set()
        self.current_reward = 0

        # Objects
        self.joe = Joe()
        self.skull = Skull()
        self.key = Object("Key", (16, 106), Task.region((16, 106)))
        self.door = Object("Door", (137, 72), Task.region((137, 72)))

        # Stochastic start
        self.env.reset()
        for delay in range(randrange(100)):
            for step in range(4):
                image, reward, done, info = self.env.step(0)
                self.skull.update(image)

        TaskInterface.__init__(self)

    def actions(self):
        """Return a set of action choices."""
        return {0, 1, 2, 3, 4, 5, 11, 12}

    def perform(self, action, render=False, delay=0):
        """Perform the given action."""
        self.current_reward = 0

        # Take four steps
        for step in range(4):
            image, reward, done, info = self.env.step(action)
            if render:
                sleep(delay)
                self.env.render()
            if info["ale.lives"] < 6:
                self.joe.dead = True
            if reward > 0:
                self.current_reward = reward

        # Update objects
        self.joe.update(image)
        self.skull.update(image)

    def update(self):
        """Override to process events after perform."""
        self.current_events = set()

        # Disambiguate deaths
        if self.joe.dead:
            while self.joe.size > 50:
                image, reward, done, info = self.env.step(0)
                self.joe.update(image)

        # Detect falls
        if self.joe.dead and self.joe.size > 20:
            self.current_events.add(Event("falls", self.joe))

        # Detect collisions
        elif self.joe.dead:
            self.current_events.add(Event("collides", self.joe, self.skull))
            self.skull.dead = True

        # Detect key acquisition
        elif self.current_reward == 100:
            self.current_events.add(Event("arrives", self.joe, self.key))
            self.door = Object("Door+Key", self.door.location, self.door.region)
            self.key = None

        # Detect success
        elif self.current_reward == 300:
            self.current_events.add(Event("arrives", self.joe, self.door))
            self.door = None

        TaskInterface.update(self)

    def objects(self):
        """Return a set of perceived objects."""
        existing = {obj for obj in {self.key, self.door} if obj is not None}
        alive = {obj for obj in {self.skull, self.joe} if not obj.dead}
        return existing | alive

    def events(self):
        """Return a set of perceived events."""
        return self.current_events

    def succeeded(self):
        """Return whether this attempt has succeeded."""
        return self.current_reward == 300

    def failed(self):
        """Return whether this attempt has failed."""
        return self.joe.dead

    def ended(self):
        """Return whether this attempt has ended."""
        return self.succeeded() or self.failed()

    def score(self):
        """Return a score for this attempt."""
        return 1 if self.succeeded() else 0

    def length(self):
        """Return a length for this attempt."""
        return len(self.record)

    def demonstrate(self):
        """Allow a person to demonstrate this task."""
        self.env.render()
        self.action = 0

        # Control actions with keys
        @self.env.viewer.window.event
        def on_key_press(symbol, modifiers):
            self.action = Task.act(symbol)

        # Default to no action
        @self.env.viewer.window.event
        def on_key_release(symbol, modifiers):
            self.action = 0

        # Perform actions on a timer
        def tick(dt):
            if not self.ended():
                self.perform(self.action, render=True, delay=SLOW)
                self.update()
                pyglet.clock.schedule_once(tick, 0)

        tick(0)
        pyglet.app.run()

    def display(self, agent):
        """Show the given agent attempting this task."""
        agent.prepare()
        self.env.render()
        while not self.ended():
            self.perform(agent.act(self, verbose=True), render=True, delay=FAST)
            self.update()
            agent.update(self)
        self.env.close()

    @staticmethod
    def act(symbol):
        """Return an action for the given key."""
        if symbol == pyglet.window.key.SPACE:
            return 1  # jump up
        elif symbol == pyglet.window.key.W:
            return 2  # move up
        elif symbol == pyglet.window.key.D:
            return 3  # move right
        elif symbol == pyglet.window.key.A:
            return 4  # move left
        elif symbol == pyglet.window.key.S:
            return 5  # move down
        elif symbol == pyglet.window.key.E:
            return 11  # jump right
        elif symbol == pyglet.window.key.Q:
            return 12  # jump left
        else:
            return 0  # noop

    @staticmethod
    def region(location):
        """Return a region label for the given location."""
        x, y = location
        if y < 89:
            return "left-door" if x < 56 else "middle-platform" if x < 102 else "right-door"
        elif y < 124:
            return "left-platform" if x < 39 else "belt" if x < 79 else "middle-ladder" if x < 81 else "belt" if x < 111 else "rope" if x < 113 else "right-platform"
        elif y < 127:
            return "left-platform" if x < 39 else "belt" if x < 111 else "rope" if x < 113 else "right-platform"
        elif y < 150:
            return "left-ladder" if x < 32 else "floor" if x < 111 else "rope" if x < 113 else "floor" if x < 128 else "right-ladder"
        elif y < 168:
            return "left-ladder" if x < 32 else "floor" if x < 128 else "right-ladder"
        else:
            return "floor"
