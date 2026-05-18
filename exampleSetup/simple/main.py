from config import DroneConfig
from quad_sim.bases.sim import NCopterBase

import numpy as np

sim = NCopterBase(agents=[DroneConfig(drone_id="drone_1")])
print(sim)
