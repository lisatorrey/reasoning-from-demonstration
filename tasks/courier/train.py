"""Train and save an agent for Courier."""

from courier import Task

from rfd.agent import Agent
from rfd.procedures import train

DEMO_FILE = "saved/demo.pkl"  # Generated by demonstrate.py
AGENT_FILE = "saved/agent.pkl"  # Created by this script

ATTEMPTS = 500  # How long to train the agent
WINDOW = 50  # Attempts averaged into each score report
FREQUENCY = 5  # Attempts between score reports

train(Agent(), lambda: Task(), DEMO_FILE, AGENT_FILE, ATTEMPTS, WINDOW, FREQUENCY)
