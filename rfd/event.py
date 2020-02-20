"""Tools for representing object-oriented events."""


class Object(object):
    """Noteworthy entity in an environment."""

    def __init__(self, object_type, location, region=None, velocity=None):
        self.type = object_type
        self.location = location
        self.region = region
        self.velocity = velocity

    def __str__(self):
        """So that objects can be printed."""
        return self.type

    def __lt__(self, other):
        """So that objects can be ordered."""
        return self.location < other.location

    def relative(self, other):
        """Return the location of this object relative to the given one."""
        return tuple(a - b for (a, b) in zip(self.location, other.location))

    def distance(self, other):
        """Return the distance between this object and the given one."""
        return sum(abs(d) for d in self.relative(other))


class Event(object):
    """Noteworthy incident involving one or two objects."""

    def __init__(self, event_type, actor, subject=None):
        self.type = event_type
        self.actor = actor
        self.subject = subject
        self.template = Template(self.type, actor.type, None if subject is None else subject.type)

    def __str__(self):
        """So that events can be printed."""
        return str(self.template)

    def __eq__(self, other):
        """So that events can be compared."""
        if not isinstance(other, Event):
            return False
        else:
            return (self.type, self.actor, self.subject) == (other.type, other.actor, other.subject)

    def __hash__(self):
        """So that events can be hashed."""
        return hash((self.type, self.actor, self.subject))

    def regional(self):
        """Return whether the objects involved in this event are within one region."""
        return self.subject is None or self.actor.region == self.subject.region

    def distance(self):
        """Return the distance between the objects involved in this event."""
        return 0 if self.subject is None else self.actor.distance(self.subject)

    def state(self):
        """Return a tuple describing the objects involved in this event."""
        if self.subject is None:
            return self.actor.location + (self.actor.velocity,)
        else:
            return self.actor.relative(self.subject) + (self.actor.velocity, self.subject.velocity)


class Template(object):
    """Description of an event in terms of its object types."""

    def __init__(self, event_type, actor_type, subject_type):
        self.type = event_type
        self.actor_type = actor_type
        self.subject_type = subject_type

    def __str__(self):
        """So that templates can be printed."""
        if self.subject_type is None:
            return self.type + "(" + self.actor_type + ")"
        else:
            return self.type + "(" + self.actor_type + ", " + self.subject_type + ")"

    def __eq__(self, other):
        """So that templates can be compared."""
        if not isinstance(other, Template):
            return False
        else:
            return (self.type, self.actor_type, self.subject_type) == (other.type, other.actor_type, other.subject_type)

    def __hash__(self):
        """So that templates can be hashed."""
        return hash((self.type, self.actor_type, self.subject_type))
