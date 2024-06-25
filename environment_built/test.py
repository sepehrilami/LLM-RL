import json
from orbit import Orbit
from datetime import datetime

start_time = datetime.now()
# Load settings
family_settings = json.load(open('settings/family_settings.json'))
orbit_settings = json.load(open('settings/orbit_settings.json'))

orbit = Orbit(orbit_settings, family_settings)


for i in range(orbit.orbit_steps):
    orbit.step()
    print('Duration: {}'.format(datetime.now() - start_time), 'iteration ', i)
