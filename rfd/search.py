"""Tools for spatial reasoning."""

from math import inf
from heapq import heappush, heappop

from rfd.event import Object, Event


class Map(object):
    """Graph of region connectivity."""

    def __init__(self):
        self.entrances = dict()

    def display(self):
        """Print the region connectivity."""
        for region in self.entrances:
            print("Region:", region)
            for neighbor in self.entrances[region]:
                print(" ->", neighbor, "at", next(iter(self.entrances[region][neighbor])))

    def update(self, frame):
        """Add region connectivity based on observed movements."""
        for obj in frame.objects:
            if frame.transitions(obj):
                current = frame.regions[obj]
                previous = frame.previous.regions[obj]
                if current is not None and previous is not None:
                    location = frame.locations[obj]
                    entrance = Object(str(location), location, current)
                    if previous not in self.entrances:
                        self.entrances[previous] = dict()
                    if current not in self.entrances[previous]:
                        self.entrances[previous][current] = {location: entrance}

    def search(self, source, targets):
        """Find shortest paths from the source to the targets."""

        # Group targets by region
        regions = dict()
        for obj in targets:
            if obj is not None:
                if obj.region not in regions:
                    regions[obj.region] = {obj}
                else:
                    regions[obj.region].add(obj)

        # Initialize search data
        predecessors = {source: None}
        distances = {source: 0}
        frontier = [(0, source)]
        done = set()

        # Traverse the graph
        while len(frontier) > 0:
            (current_cost, current) = heappop(frontier)
            if current not in done:
                done.add(current)

                # Improve paths to targets in this region
                if current.region in regions:
                    for obj in regions[current.region]:
                        obj_cost = current_cost + obj.distance(current)
                        if obj not in distances or distances[obj] > obj_cost:
                            predecessors[obj] = current
                            distances[obj] = obj_cost

                # Check neighboring regions
                if current.region in self.entrances:
                    for next_region in self.entrances[current.region]:
                        for entrance in self.entrances[current.region][next_region].values():
                            entrance_cost = current_cost + entrance.distance(current)

                            # Improve paths to entrances in this region
                            if entrance not in distances or distances[entrance] > entrance_cost:
                                predecessors[entrance] = current
                                distances[entrance] = entrance_cost
                                heappush(frontier, (entrance_cost, entrance))

        return Search(predecessors, distances)


class Search(object):
    """Result of a map search."""

    def __init__(self, predecessors, distances):
        self.predecessors = predecessors
        self.distances = distances

    def found(self, obj):
        """Return whether this search contains a path to the given object."""
        return obj in self.predecessors

    def distance(self, obj):
        """Return the distance to the given object in this search."""
        if obj is None:
            return 0
        elif obj in self.distances:
            return self.distances[obj]
        else:
            return inf

    def checkpoint(self, objective):
        """Return the first checkpoint on the path to the given objective."""
        checkpoint = objective.subject
        while self.predecessors[checkpoint] != objective.actor:
            checkpoint = self.predecessors[checkpoint]
        return Event("checkpoint", objective.actor, checkpoint)
