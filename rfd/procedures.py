"""Common procedures for training and inspecting agents."""

from pickle import load


def train(agent, task_generator, demo_file, agent_file, attempts, window, frequency):
    """Train and save one agent."""

    f = open(demo_file, "rb")
    demo = load(f)
    f.close()

    agent.observe(demo)
    agent.save(agent_file)

    lengths = list()
    scores = list()

    for attempt in range(1, attempts + 1):
        task = task_generator()
        agent.attempt(task)

        lengths.append(task.length())
        scores.append(task.score())

        if attempt % frequency == 0:
            length = sum(lengths)
            score = sum(scores[-window:]) / len(scores[-window:])

            agent.save(agent_file)
            print("{:<10}{:<10}{:<10}".format(attempt, round(score, 2), length))


def display(task_generator, agent_file):
    """Show a saved agent making attempts."""

    f = open(agent_file, "rb")
    agent = load(f)
    f.close()
    print()

    input("Enter to view theory:")
    agent.theory.display()
    print()

    input("Enter to view map:")
    agent.map.display()
    print()

    while True:
        input("Enter to attempt task:")
        task = task_generator()
        task.display(agent)

        print("Length:", task.length())
        print("Score:", task.score())
        print()


def plot(agent_generator, task_generator, demo_file, plot_file, curves, attempts, window, frequency):
    """Plot multiple learning curves."""

    f = open(demo_file, "rb")
    record = load(f)
    f.close()

    f = open(plot_file, "w")
    f.close()

    for curve in range(1, curves + 1):
        print("Curve", curve, "...")

        agent = agent_generator()
        agent.observe(record)

        lengths = list()
        scores = list()

        for attempt in range(1, attempts + 1):
            task = task_generator()
            agent.attempt(task)

            lengths.append(task.length())
            scores.append(task.score())

            if attempt % frequency == 0:
                length = sum(lengths)
                score = sum(scores[-window:]) / len(scores[-window:])

                f = open(plot_file, "a")
                f.write(str(sum(lengths)) + "\t"*curve + str(score) + "\n")
                f.close()
