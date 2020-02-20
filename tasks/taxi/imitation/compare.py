"""Compare RfD with imitation learning in Taxi."""

from imitation import Driver, Comparison, Plot

PLOT_FILE = "saved/plot.txt"  # Created by this script

REPEATS = 10  # Number of agents of each type
LEVELS = [1, 3, 9]  # Numbers of demonstrations to provide

comparisons = list()
for repeat in range(1, REPEATS + 1):
    print("Repeat", repeat, "...")
    comparisons.append(Comparison(LEVELS))

plot = Plot()
plot.add([comparison.independent for comparison in comparisons])

for level in LEVELS:
    plot.add([comparison.imitators[level] for comparison in comparisons])

for level in LEVELS:
    plot.add([comparison.decomposers[level] for comparison in comparisons])

plot.save(PLOT_FILE)
