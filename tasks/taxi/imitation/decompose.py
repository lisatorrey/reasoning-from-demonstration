"""Measure the number of demos needed for ADA to find the optimal decomposition for Taxi."""

from imitation import Driver

from ada import decompose

TRIALS = 100  # Number of independent trials
REPEATS = 16  # Decomposition attempts with each group of demos
MAX_DEMOS = 32  # Demonstrations performed before giving up on a trial

successes = list()

for trial in range(1, TRIALS + 1):
    print("Trial", trial, "...")
    
    driver = Driver()
    driver.train()

    num_demos = 0
    demos = dict()
    optimal = False

    while not optimal and num_demos < MAX_DEMOS:
        driver.generate(demos)
        num_demos += 1
        repeat = 0
        
        while not optimal and repeat < REPEATS:
            optimal = decompose(demos)
            repeat += 1

    if optimal:
        successes.append(num_demos)

print("Number of successes:", len(successes), "/", TRIALS)
print("Demos before success:", sorted(successes))
