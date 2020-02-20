"""Task defined in Montezuma's Revenge from OpenAI Gym, but with extended actions."""

from time import sleep

from tasks.montezuma.montezuma import Joe as OriginalJoe
from tasks.montezuma.montezuma import Task as OriginalTask


class Joe(OriginalJoe):
    def update(self, image):
        """Override to replace velocity with fall tracking."""
        OriginalJoe.update(self, image)
        self.falling = self.velocity[1] > 0
        self.velocity = None


class Task(OriginalTask):
    def __init__(self):
        """Override to replace Joe and track steps."""
        OriginalTask.__init__(self)
        self.joe = Joe()
        self.steps = 0

    def perform(self, action, render=False, delay=0):
        """Override to continue actions until Joe has control again."""
        self.current_reward = 0

        # Initial four steps
        for step in range(4):
            self.step(action, render, delay)

        # Complete a jump
        for step in range(12):
            if action in {1, 11, 12} and not self.joe.dead:
                self.step(0, render, delay)

        # Complete a fall
        while self.joe.falling and not self.joe.dead:
            self.step(0, render, delay)

    def step(self, action, render=False, delay=0):
        """Take one step."""
        image, reward, done, info = self.env.step(action)
        if render:
            sleep(delay)
            self.env.render()
        if info["ale.lives"] < 6:
            self.joe.dead = True
        if reward > 0:
            self.current_reward = reward

        # Update objects
        self.steps += 1
        self.joe.update(image)
        if self.steps % 4 == 0:
            self.skull.update(image)

    def length(self):
        """Override to keep timing comparable."""
        return self.steps // 4
