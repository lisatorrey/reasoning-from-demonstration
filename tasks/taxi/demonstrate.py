"""Create and save a demonstration of Taxi."""

from taxi import Task

DEMO_FILE = "saved/demo.pkl"  # Created by this script

task = Task()
task.demonstrate()
task.save(DEMO_FILE)
