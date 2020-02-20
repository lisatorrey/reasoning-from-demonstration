"""Tools for defining and recording tasks."""

from pickle import dump

from rfd.event import Event


class TaskInterface(object):
    """Superclass for task definitions."""

    def __init__(self):
        self.record = list()
        self.frame = Frame(None, self.objects(), set(), False, False)

    def save(self, filename):
        """Save a record of this attempt."""
        try:
            f = open(filename, "wb")
        except FileNotFoundError:
            filename = input("Need another demo filename: ")
            self.save(filename)
            return
        try:
            dump(self.record, f)
        except MemoryError:
            dump("MemoryError", f)
        f.close()

    def update(self):
        """Add to the record of this attempt."""
        self.frame = Frame(self.frame, self.objects(), self.events(), self.succeeded(), self.failed())
        self.record.append(self.frame)

    def actions(self):
        """Return a set of action choices."""
        raise NotImplementedError

    def perform(self, action):
        """Perform the given action."""
        raise NotImplementedError

    def objects(self):
        """Return a set of perceived objects."""
        raise NotImplementedError

    def events(self):
        """Return a set of perceived events."""
        raise NotImplementedError

    def succeeded(self):
        """Return whether this attempt has succeeded."""
        raise NotImplementedError

    def failed(self):
        """Return whether this attempt has failed."""
        raise NotImplementedError

    def ended(self):
        """Return whether this attempt has ended."""
        raise NotImplementedError

    def score(self):
        """Return a score for this attempt."""
        raise NotImplementedError

    def length(self):
        """Return a length for this attempt."""
        raise NotImplementedError

    def demonstrate(self):
        """Allow a person to demonstrate this task."""
        raise NotImplementedError

    def display(self, agent):
        """Show the given agent attempting this task."""
        raise NotImplementedError


class Frame(object):
    """Recorded step in a task attempt."""

    def __init__(self, previous, objects, events, success, failure):
        self.previous = previous

        # Object record
        self.objects = objects
        self.regions = {obj: obj.region for obj in objects}
        self.locations = {obj: obj.location for obj in objects}

        # Event record
        self.events = events
        self.success = success
        self.failure = failure

    def observations(self):
        """Return a set of any appearances and endings that occurred in this frame."""
        observations = set()

        # Appearances
        if self.previous is not None:
            for obj in self.objects:
                if obj not in self.previous.objects:
                    observations.add(obj.type)

        # Endings
        if self.success:
            observations.add("SUCCESS")
        if self.failure:
            observations.add("FAILURE")

        return observations

    def transitions(self, obj):
        """Return whether the given object changed its region in this frame."""
        return obj in self.objects and obj in self.previous.objects and self.regions[obj] != self.previous.regions[obj]

    def supports(self, objective):
        """Return whether the given objective is pursuable in this frame."""
        if objective.subject is None:
            return objective.actor in self.objects
        else:
            return {objective.actor, objective.subject} <= self.objects

    def objectives(self, template):
        """Return a set of objectives that match the given event template."""      
        actors = {obj for obj in self.objects if obj.type == template.actor_type}
        if template.subject_type is None:
            return {Event(template.type, actor, None) for actor in actors}
        else:
            subjects = {obj for obj in self.objects if obj.type == template.subject_type}
            return {Event(template.type, a, s) for a in actors for s in subjects if a != s}

    def missing(self, template):
        """Return a set of object types required for the given event template that are missing in this frame."""
        missing = [template.actor_type]
        if template.subject_type is not None:
            missing.append(template.subject_type)
        for obj in self.objects:
            if obj.type in missing:
                missing.remove(obj.type)
        return set(missing)
