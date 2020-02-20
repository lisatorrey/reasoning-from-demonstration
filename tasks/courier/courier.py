"""Task in which a courier dodges vehicles to collect and deliver packages."""

from random import choice, random

from tkinter import Tk, Frame, Canvas

from rfd.task import TaskInterface
from rfd.event import Event, Object

# Counts
ROWS = 35
COLS = 35
PACKAGES = 4
VEHICLES = 20

# Quadrants
ALL = -1
CENTER = 0
TOP_LEFT = 1
TOP_RIGHT = 2
BOTTOM_LEFT = 3
BOTTOM_RIGHT = 4

SCALE = 16  # Pixels per grid cell in the environment view
SLOW = 200  # Milliseconds between renders for playability
FAST = 50  # Milliseconds between renders for watchability


class Courier(Object):
    def __init__(self, location, packages=0):
        self.dead = False
        self.packages = packages
        object_type = "Courier" + ("" if packages == 0 else "+"+str(packages))
        Object.__init__(self, object_type, location, region=Environment.region(location))


class Platform(Object):
    def __init__(self, location, packages=0):
        self.packages = packages
        object_type = "Platform" + ("" if packages == 0 else "+"+str(packages))
        Object.__init__(self, object_type, location, region=Environment.region(location))


class Package(Object):
    def __init__(self, location):
        Object.__init__(self, "Package", location, region=Environment.region(location))


class Vehicle(Object):
    def __init__(self, location, direction):
        Object.__init__(self, "Vehicle", location, region=Environment.region(location), velocity=direction)


class Environment(object):
    def __init__(self):
        center = (ROWS // 2, COLS // 2)
        self.courier = Courier(center)
        self.platform = Platform(center)

        # Set boundaries
        self.walls = {(r, 0) for r in range(ROWS)}
        self.walls |= {(0, c) for c in range(COLS)}
        self.walls |= {(r, COLS - 1) for r in range(ROWS)}
        self.walls |= {(ROWS - 1, c) for c in range(COLS)}
        self.walls |= {(r, COLS // 3) for r in range(0, ROWS*2 // 3)}
        self.walls |= {(r, COLS*2 // 3) for r in range(ROWS // 3, ROWS)}

        # Distribute packages
        spots = [(r, c) for r in range(1, ROWS*2 // 3) for c in range(2, COLS//3 - 1)]
        spots += [(r, c) for r in range(1 + ROWS // 3, ROWS - 1) for c in range(2 + COLS*2 // 3, COLS - 2)]

        self.packages = set()
        while len(self.packages) < PACKAGES:
            location = choice(spots)
            spots.remove(location)
            self.packages.add(Package(location))

        # Distribute vehicles
        spots += [(r, c) for r in range(ROWS*2 // 3, ROWS - 1) for c in range(2, COLS//3 - 1)]
        spots += [(r, c) for r in range(1, 1 + ROWS // 3) for c in range(2 + COLS*2 // 3, COLS - 2)]

        self.vehicles = set()
        while len(self.vehicles) < VEHICLES:
            location = choice(spots)
            spots.remove(location)
            self.vehicles.add(Vehicle(location, "up" if location[1] < COLS // 3 else "down"))

    def step(self, action):
        """Execute the given action and return a set of events."""
        events = set()

        # Player movement
        if not self.courier.dead:
            location = Environment.neighbor(self.courier.location, action)
            if location not in self.walls:
                self.courier.location = location
                self.courier.region = Environment.region(location)
                for vehicle in self.vehicles:
                    if location == vehicle.location:
                        events.add(Event("collides", self.courier, vehicle))
                        self.courier.dead = True
                        return events

        # Vehicle movement
        for vehicle in self.vehicles:
            vehicle.location = self.neighbor(vehicle.location, vehicle.velocity)
            if vehicle.location[0] < 0:
                vehicle.location = (ROWS - 1, vehicle.location[1])
            elif vehicle.location[0] == ROWS:
                vehicle.location = (0, vehicle.location[1])
            elif vehicle.location == self.courier.location:
                events.add(Event("collides", self.courier, vehicle))
                self.courier.dead = True
                return events

        # Package collection
        for package in self.packages.copy():
            if package.location == self.courier.location:
                events.add(Event("arrives", self.courier, package))
                self.courier = Courier(self.courier.location, self.courier.packages + 1)
                self.packages.remove(package)

        # Package delivery
        if self.courier.location == self.platform.location:
            events.add(Event("arrives", self.courier, self.platform))
            if self.courier.packages > 0:
                self.platform = Platform(self.platform.location, self.platform.packages + self.courier.packages)
                self.courier = Courier(self.courier.location)
        
        return events

    @staticmethod
    def neighbor(location, direction):
        """Return the neighbor of the given location in the given direction."""
        if direction == "up":
            return (location[0] - 1, location[1])
        elif direction == "down":
            return (location[0] + 1, location[1])
        elif direction == "left":
            return (location[0], location[1] - 1)
        elif direction == "right":
            return (location[0], location[1] + 1)
        else:
            return location

    @staticmethod
    def region(location):
        """Return a region label for the given location."""
        if location[1] < COLS // 3:
            return "left"
        elif location[1] > COLS*2 // 3:
            return "right"
        else:
            return "middle"


class View(Frame):
    def __init__(self):
        self.action = None
        Frame.__init__(self, master=Tk())
        self.master.bind("<Key>", self.onkey)
        self.master.attributes("-topmost", True)
        self.canvas = Canvas(self, width=SCALE*COLS + 1, height=SCALE*ROWS + 1)
        self.canvas.pack()
        self.pack()

    def onkey(self, event):
        """Allow keys to change the player's direction."""
        if event.keysym == 'w':
            self.action = "up"
        elif event.keysym == 'a':
            self.action = "left"
        elif event.keysym == 's':
            self.action ="down"
        elif event.keysym == 'd':
            self.action = "right"
        else:
            self.action = None

    def render(self, env):
        """Display the given environment."""
        self.canvas.delete("all")
        
        for location in env.walls:
            self.canvas.create_rectangle(*View.box(location), fill="black")
        
        self.canvas.create_rectangle(*View.box(env.platform.location), fill="white")
        for i in range(env.platform.packages):
            self.canvas.create_rectangle(*View.box(env.platform.location, i + 1), fill="light blue")
        
        self.canvas.create_oval(*View.box(env.courier.location), fill="yellow")
        for i in range(env.courier.packages):
            self.canvas.create_rectangle(*View.box(env.courier.location, i + 1), fill="light blue")

        for package in env.packages:
            self.canvas.create_rectangle(*View.box(package.location), fill="light blue")

        for vehicle in env.vehicles:
            self.canvas.create_polygon(*View.trapezoid(vehicle.location, vehicle.velocity), fill="red")

    @staticmethod
    def box(location, quadrant=ALL):
        """Return canvas coordinates for a bounding box at the given location and quadrant."""
        x1 = 2 + SCALE * location[1]
        y1 = 2 + SCALE * location[0]

        if quadrant in (TOP_RIGHT, BOTTOM_RIGHT):
            x1 += SCALE // 2
        if quadrant in (BOTTOM_LEFT, BOTTOM_RIGHT):
            y1 += SCALE // 2
        if quadrant == CENTER:
            x1 += SCALE // 4
            y1 += SCALE // 4

        if quadrant == ALL:
            x2 = x1 + SCALE
            y2 = y1 + SCALE
        else:
            x2 = x1 + SCALE // 2
            y2 = y1 + SCALE // 2

        return x1 + 1, y1 + 1, x2 - 1, y2 - 1

    @staticmethod
    def trapezoid(location, direction):
        """Return canvas coordinates for a trapezoid at the given location and direction."""
        x1 = x4 = 2 + SCALE * location[1]
        y1 = y2 = 2 + SCALE * location[0]
        x2 = x3 = 2 + SCALE * (location[1] + 1)
        y3 = y4 = 2 + SCALE * (location[0] + 1)

        if direction == "up":
            x1 += SCALE // 4
            x2 -= SCALE // 4
        elif direction == "down":
            x4 += SCALE // 4
            x3 -= SCALE // 4

        return x1 + 1, y1 + 1, x2 - 1, y2 + 1, x3 - 1, y3 - 1, x4 + 1, y4 - 1


class Task(TaskInterface):
    def __init__(self):
        self.env = Environment()
        self.current_events = set()
        TaskInterface.__init__(self)

    def actions(self):
        """Return a set of action choices."""
        return {None, "up", "down", "left", "right"}

    def perform(self, action):
        """Perform the given action."""
        self.current_events = self.env.step(action)

    def objects(self):
        """Return a set of perceived objects."""
        return self.env.packages | self.env.vehicles | {self.env.platform, self.env.courier}

    def events(self):
        """Return a set of perceived events."""
        return self.current_events

    def succeeded(self):
        """Return whether this attempt has succeeded."""
        return self.env.platform.packages == PACKAGES

    def failed(self):
        """Return whether this attempt has failed."""
        return self.env.courier.dead
    
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
        view = View()
        view.render(self.env)

        # Perform actions on a timer
        def tick():
            if not self.ended():
                self.perform(view.action)
                self.update()
                view.render(self.env)
                view.after(SLOW, tick)

        view.after(SLOW, tick)
        view.mainloop()

    def display(self, agent):
        """Show the given agent attempting this task."""
        agent.prepare()
        view = View()
        view.render(self.env)

        # Perform actions on a timer
        def tick():
            if not self.ended():
                self.perform(agent.act(self, verbose=True))
                self.update()
                agent.update(self)
                view.render(self.env)
                view.after(FAST, tick)

        view.after(FAST, tick)
        view.mainloop()
