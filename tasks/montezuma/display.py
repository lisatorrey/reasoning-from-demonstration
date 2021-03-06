"""Show a saved agent attempting Montezuma's Revenge."""

from montezuma import Task

from rfd.procedures import display

AGENT_FILE = "saved/agent.pkl"  # Created by train.py

display(lambda: Task(), AGENT_FILE)
