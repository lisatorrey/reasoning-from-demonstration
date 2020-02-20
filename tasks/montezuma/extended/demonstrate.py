"""Create and save a demonstration of Montezuma's Revenge with extended actions."""

from montezuma import Task

DEMO_FILE = "saved/demo.pkl"  # Created by this script

task = Task()
task.demonstrate()
task.save(DEMO_FILE)
