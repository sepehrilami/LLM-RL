import json
from orbit import Orbit, agent_individual
from datetime import datetime
import networkx as nx
import numpy as np

start_time = datetime.now()
# Load settings
family_settings = json.load(open('settings/family_settings.json'))
orbit_settings = json.load(open('settings/orbit_settings.json'))

orbit = Orbit(orbit_settings)
agent_test = agent_individual(orbit, 'agent_test', family_settings, 0).agent_initialization()
observation_ind = [0, 1]
agent_test.step(observation_ind)