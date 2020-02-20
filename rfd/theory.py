"""Tools for causal reasoning."""


class Theory(object):
    """Causal model of events in a task."""

    def __init__(self):
        self.hypotheses = dict()
        self.experienced = set()

    def display(self):
        """Print the causal model."""
        for effect in self.hypotheses:
            if len(self.hypotheses[effect]) > 0:
                print("Causes of", str(effect))
                for cause in self.hypotheses[effect]:
                    print("\t" + str(cause))

    def update(self, frame):
        """Adjust hypotheses based on the given frame."""
        causes = {event.template for event in frame.events}
        effects = frame.observations()
        self.expand(causes, effects)
        self.contract(causes, effects)

    def expand(self, causes, effects):
        """Add plausible hypotheses."""
        for cause in causes:
            if cause not in self.experienced:
                self.experienced.add(cause)
                for effect in effects:
                    if effect not in self.hypotheses:
                        self.hypotheses[effect] = {cause}
                    else:
                        self.hypotheses[effect].add(cause)

    def contract(self, causes, effects):
        """Remove implausible hypotheses."""
        for effect in self.hypotheses:
            for cause in causes:
                if cause in self.hypotheses[effect] and effect not in effects:
                    self.hypotheses[effect].remove(cause)

    def causes(self, effect):
        """Return a set of hypothesized causes for the given effect."""
        return self.hypotheses[effect] if effect in self.hypotheses else set()

    def contributors(self, effect, frame, ancestors=None):
        """Return a set of event templates that contribute towards the given effect."""
        templates = set()
        ancestors = set() if ancestors is None else ancestors
        for cause in self.causes(effect):
            missing = frame.missing(cause)
            if len(missing) == 0:
                templates.add(cause)
            elif len(missing & ancestors) == 0:
                for object_type in missing:
                    templates |= self.contributors(object_type, frame, ancestors | {effect})
        return templates
