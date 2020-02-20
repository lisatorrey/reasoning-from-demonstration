"""Task defined in Ms. Pacman from OpenAI Gym."""

import gym
import pyglet
import numpy as np

from time import sleep
from random import randrange

from rfd.task import TaskInterface
from rfd.event import Object, Event

SLOW = 0.03  # Seconds between renders for playability
FAST = 0.01  # Seconds between renders for watchability


class Pacman(Object):
    def __init__(self):
        self.dead = False
        Object.__init__(self, "Pacman", (77, 103), region=Task.region((77, 103)))

    def update(self, x, y):
        """Process new coordinates."""
        dx = x - self.location[0]
        dy = y - self.location[1]
        self.location = (x, y)
        self.velocity = Task.velocity(dx, dy)
        self.region = Task.region((x, y), self.velocity)


class Ghost(Object):
    def __init__(self, location, color, small=False):
        self.color = color
        object_type = "Edible" if color == 66 else "Eyes" if small else "Ghost"
        Object.__init__(self, object_type, location, region=Task.region(location))

    def update(self, new):
        """Process new properties."""
        x, y = new.location
        dx = x - self.location[0]
        dy = y - self.location[1]
        self.location = (x, y)
        self.velocity = Task.velocity(dx, dy, self.velocity)
        self.region = Task.region((x, y), self.velocity)


class Task(TaskInterface):
    def __init__(self):
        self.env = gym.make("MsPacmanNoFrameskip-v4")
        self.previous_ghosts = set()
        self.current_events = set()
        self.image_buffer = list()
        self.current_reward = 0
        self.success_count = 0

        # Objects
        self.pacman = Pacman()
        self.ghosts = {
            Ghost((79, 55), 200),
            Ghost((80, 85), 198),
            Ghost((79, 85), 180),
            Ghost((81, 85), 84)
        }
        self.powers = {
            Object("Power", (9, 18), region=Task.region((9, 18))),
            Object("Power", (149, 17), region=Task.region((149, 17))),
            Object("Power", (9, 150), region=Task.region((9, 150))),
            Object("Power", (149, 149), region=Task.region((149, 149)))
        }

        # Built-in delay
        self.env.reset()
        for step in range(266):
            self.env.step(0)

        # Stochastic start
        for delay in range(randrange(25)):
            self.perform(0)

        TaskInterface.__init__(self)

    def actions(self):
        """Return a set of action choices."""
        return {1, 2, 3, 4}

    def perform(self, action, render=False, delay=0):
        """Perform the given action."""
        self.current_events = set()
        self.current_reward = 0

        # Take four steps
        for step in range(4):
            image, reward, done, info = self.env.step(action)
            self.image_buffer.append(image)
            if render:
                sleep(delay)
                self.env.render()
            if reward >= 200:
                self.success_count += 1
            if reward >= 50:
                self.current_reward = reward

        # Update objects
        self.image_buffer = self.image_buffer[-5:]
        self.analyze()

        # Detect death
        if self.pacman.dead:
            ghosts = {ghost for ghost in self.previous_ghosts if ghost.type == "Ghost"}
            if len(ghosts) > 0:  # In case of tracking errors
                ghost = sorted(ghosts, key=lambda g: g.distance(self.pacman))[0]
                if ghost.distance(self.pacman) < 12:  # In case of tracking errors
                    self.current_events.add(Event("collides", self.pacman, ghost))

        # Detect powerup
        if self.current_reward % 100 == 50:
            self.ghosts.add(Ghost((75,75), 66))  # In case of tracking errors
            power = sorted(self.powers, key=lambda p: p.distance(self.pacman))[0]
            self.current_events.add(Event("arrives", self.pacman, power))
            self.powers.remove(power)

        # Detect success
        if self.current_reward >= 200:
            edibles = {ghost for ghost in self.previous_ghosts if ghost.type == "Edible"}
            if len(edibles) > 0:  # In case of tracking errors
                edible = sorted(edibles, key=lambda g: g.distance(self.pacman))[0]
                if edible.distance(self.pacman) < 12:  # In case of tracking errors
                    self.current_events.add(Event("catches", self.pacman, edible))

    def objects(self):
        """Return a set of perceived objects."""
        return self.powers | self.ghosts | {obj for obj in {self.pacman} if not obj.dead}

    def events(self):
        """Return a set of perceived events."""
        return self.current_events

    def succeeded(self):
        """Return whether this attempt has succeeded."""
        return self.current_reward >= 200

    def failed(self):
        """Return whether this attempt has failed."""
        return self.pacman.dead

    def ended(self):
        """Return whether this attempt has ended."""
        return self.failed()

    def score(self):
        """Return a score for this attempt."""
        return self.success_count

    def length(self):
        """Return a length for this attempt."""
        return len(self.record)

    def demonstrate(self):
        """Allow a person to demonstrate this task."""
        self.env.render()
        self.action = 3

        # Control actions with keys
        @self.env.viewer.window.event
        def on_key_press(symbol, modifiers):
            action = Task.act(symbol)
            if action is not None:
                self.action = action

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

    def analyze(self):
        """Find sprites in the image buffer."""
        self.previous_ghosts = self.ghosts
        self.ghosts = set()

        # Find pacman
        for img in reversed(self.image_buffer):
            indices = np.where(img[:172, :, 0] == 210)
            if len(indices[0]) > 0:
                xmin, xmax, ymin, ymax = Task.bounds(sorted(zip(indices[1], indices[0])))

                # Stabilize width and height
                if np.count_nonzero(indices[1] == xmax) == 1:
                    xmin, ymax = xmax - 7, ymin + 9
                else:
                    xmax, ymin = xmin + 7, ymax - 9

                # Locate pacman
                x, y = Task.snap((xmin + xmax) // 2, (ymin + ymax) // 2)
                self.pacman.update(x, y)
                break

        # Find the dangerous ghosts
        for color in [200, 198, 180, 84]:
            for img in reversed(self.image_buffer):
                indices = np.where(img[:172, :, 0] == color)
                if len(indices[0]) > 0:
                    xmin, xmax, ymin, ymax = Task.bounds(sorted(zip(indices[1], indices[0])))
                    x, y = Task.snap((xmin + xmax) // 2, (ymin + ymax) // 2)
                    found = Ghost((x, y), color, small=len(indices[0]) < 20)

                    # Keep existing dangerous ghosts
                    placed = False
                    for ghost in self.previous_ghosts:
                        if ghost.color == found.color and ghost.type == found.type:
                            self.ghosts.add(ghost)
                            ghost.update(found)
                            placed = True
                            break

                    # Create new dangerous ghosts
                    if not placed:
                        self.ghosts.add(found)
                        break

        # Find edible ghost sprites
        sprites = list()
        for img in self.image_buffer:
            indices = np.where((img[:172, :, 0] == 66) | (img[:172, :, 0] == 214))
            if len(indices[0]) > 0:
                pixels = sorted(zip(indices[0], indices[1]))
                while len(pixels) > 0:

                    # Split vertical gaps
                    i = 1
                    while i < len(pixels) and pixels[i][0] - pixels[i-1][0] < 5:
                        i += 1

                    # Locate ghosts
                    xmin, xmax, ymin, ymax = Task.bounds(sorted((x, y) for (y, x) in pixels[:i]))
                    x, y = Task.snap((xmin + xmax) // 2, (ymin + ymax) // 2)
                    sprites.append(Ghost((x, y), 66))
                    pixels = pixels[i:]

        # Group edible ghost sprites
        clusters = list()
        for sprite in sprites:
            placed = False
            for cluster in clusters:
                if sprite.distance(cluster[-1]) < 5:
                    cluster.append(sprite)
                    placed = True
                    break
            if not placed:
                clusters.append([sprite])

        # Keep existing edible ghosts
        for ghost in self.previous_ghosts:
            if ghost not in self.ghosts and len(clusters) > 0:
                cluster = sorted(clusters, key=lambda c: c[0].distance(ghost))[0]
                first, last = cluster[0], cluster[-1]
                if ghost.type == first.type and ghost.distance(first) < 5:
                    ghost.update(last)
                    self.ghosts.add(ghost)
                    clusters.remove(cluster)

        # Create new edible ghosts
        for cluster in clusters:
            self.ghosts.add(cluster[-1])

        # Detect death
        if len(self.ghosts) == 0:
            self.pacman.dead = True

    @staticmethod
    def act(symbol):
        """Return an action for the given key."""
        if symbol == pyglet.window.key.W:
            return 1  # up
        elif symbol == pyglet.window.key.D:
            return 2  # right
        elif symbol == pyglet.window.key.A:
            return 3  # left
        elif symbol == pyglet.window.key.S:
            return 4  # down
        else:
            return None

    @staticmethod
    def velocity(dx, dy, default=None):
        """Translate velocities into corridor directions."""
        if abs(dx) > abs(dy):
            return "left" if dx < 0 else "right"
        elif abs(dy) > abs(dx):
            return "up" if dy < 0 else "down"
        else:
            return default

    @staticmethod
    def bounds(pixels):
        """Return a bounding box for the given pixels."""
        for j in range(1, len(pixels)):
            if pixels[j][0] - pixels[j-1][0] > 10:
                left, right = pixels[:j], pixels[j:]

                # Join split sprites
                if len(left) > len(right):
                    right = [(x-160, y) for (x, y) in right]
                else:
                    left = [(x+160, y) for (x, y) in left]
                pixels = left + right
                break

        xmin, xmax = min(x for (x, y) in pixels), max(x for (x, y) in pixels)
        ymin, ymax = min(y for (x, y) in pixels), max(y for (x, y) in pixels)
        return xmin, xmax, ymin, ymax

    @staticmethod
    def snap(x, y):
        """Clean up locations by snapping them to corridors when possible."""
        horizontals = {9, 17, 33, 49, 57, 65, 93, 101, 109, 125, 141, 149}
        verticals = {7, 31, 55, 79, 103, 127, 139, 163}

        if 59 <= x <= 99 and 57 <= y <= 101:
            return x, y
        elif x in horizontals or y in verticals:
            return x, y
        elif y+1 in verticals:
            return x, y+1
        elif y-1 in verticals:
            return x, y-1
        elif x+1 in horizontals:
            return x+1, y
        elif x-1 in horizontals:
            return x-1, y
        else:
            return x, y

    @staticmethod
    def region(location, velocity=None):
        """Return a region label for the given location and velocity."""
        x, y = location

        # Vertical corridors
        if velocity in {"up", "down", None}:
            if x == 9 and 127 <= y <= 163: return "V1a"
            if x == 9 and 7 <= y <= 31: return "V1b"
            if x == 17 and 31 <= y <= 127: return "V2"
            if x == 33 and 103 <= y <= 163: return "V3a"
            if x == 33 and 7 <= y <= 55: return "V3b"
            if x == 49 and 103 <= y <= 139: return "V4a"
            if x == 49 and 7 <= y <= 31: return "V4b"
            if x == 57 and 31 <= y <= 103: return "V5"
            if x == 65 and 103 <= y <= 163: return "V6"
            if x == 93 and 103 <= y <= 163: return "V7"
            if x == 101 and 31 <= y <= 103: return "V8"
            if x == 109 and 103 <= y <= 139: return "V9a"
            if x == 109 and 7 <= y <= 31: return "V9b"
            if x == 125 and 103 <= y <= 163: return "V10a"
            if x == 125 and 7 <= y <= 55: return "V10b"
            if x == 141 and 31 <= y <= 127: return "V11"
            if x == 149 and 127 <= y <= 163: return "V12a"
            if x == 149 and 7 <= y <= 31: return "V12b"

        # Horizontal corridors
        if velocity in {"right", "left", None}:
            if y == 7 and 9 <= x <= 33: return "H1a"
            if y == 7 and 49 <= x <= 109: return "H1b"
            if y == 7 and 125 <= x <= 149: return "H1c"
            if y == 31 and 9 <= x <= 149: return "H2"
            if y == 55 and x <= 17: return "H3a"
            if y == 55 and 33 <= x <= 125: return "H3b"
            if y == 55 and x >= 141: return "H3c"
            if y == 79 and 17 <= x <= 57: return "H4a"
            if y == 79 and 101 <= x <= 141: return "H4b"
            if y == 103 and x <= 17: return "H5a"
            if y == 103 and 33 <= x <= 125: return "H5b"
            if y == 103 and x >= 141: return "H5c"
            if y == 127 and 9 <= x <= 33: return "H6a"
            if y == 127 and 65 <= x <= 93: return "H6b"
            if y == 127 and 125 <= x <= 149: return "H6c"
            if y == 139 and 49 <= x <= 65: return "H7a"
            if y == 139 and 93 <= x <= 109: return "H7b"
            if y == 163 and 9 <= x <= 149: return "H8"
