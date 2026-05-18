from abc import ABC, abstractmethod

from quad_sim.bases.dynamics            import      DynamicsBase
from quad_sim.bases.controller          import      ControllerBase
from quad_sim.bases.pilot               import      PilotBase
from quad_sim.bases.allocator           import      AllocatorBase
from quad_sim.bases.integrator           import     IntegratorBase
from quad_sim.bases.setpoints           import      Setpoints
from quad_sim.bases.environment         import      EnvironmentBase
from quad_sim.bases.constraint            import      ConstraintBase


from quad_sim.bases.state                    import      StateVector



class DroneBase(ABC):

    def __init__(
                    self, drone_id: str, init_state:StateVector, pilot: PilotBase,
                    dynamics: DynamicsBase, allocator: AllocatorBase,
                    controller: ControllerBase, integrator: IntegratorBase,
                    environment: EnvironmentBase, constraints: ConstraintBase
                ):
        
        if not isinstance(drone_id, str):
            raise TypeError(f"drone_id must be a string, got {type(drone_id)}")
        if not isinstance(init_state, StateVector):
            raise TypeError(f"init_state must be a StateVector, got {type(init_state)}")
        if not isinstance(pilot, PilotBase):
            raise TypeError(f"pilot must be a PilotBase subclass, got {type(pilot)}")
        if not isinstance(dynamics, DynamicsBase):
            raise TypeError(f"dynamics must be a DynamicsBase subclass, got {type(dynamics)}")
        if not isinstance(allocator, AllocatorBase):
            raise TypeError(f"allocator must be a AllocatorBase subclass, got {type(allocator)}")
        if not isinstance(controller, ControllerBase):
            raise TypeError(f"controller must be a ControllerBase subclass, got {type(controller)}")
        if not isinstance(integrator, IntegratorBase):
            raise TypeError(f"integrator must be a IntegratorBase subclass, got {type(integrator)}")
        if not isinstance(environment, EnvironmentBase):
            raise TypeError(f"environment must be a EnvironmentBase subclass, got {type(environment)}")
        if not isinstance(constraints, ConstraintBase):
            raise TypeError(f"constraints must be a ConstraintBase subclass, got {type(constraints)}")

        self.iD = drone_id
        self.state = init_state
        self.pilot = pilot
        self.model = dynamics
        self.allocator = allocator
        self.controller = controller
        self.integrator = integrator
        self.environment = environment
        self.constraints = constraints


    def step(self) -> None:
        """
        Advances the simulation by one time step.
        This method updates the state of the drone model, applying control inputs,
        updating physics, and recalculating sensor readings as necessary.
        """
        self.target = self.get_setpoints()
        self.target = self.constraints.enforce_setpoint_constraints(self.target)
        self.response = self.pilot.compute_control(self.state, self.target)
        self.model.set_motor_rpm(self.allocator.allocate(self.response))
        
        self.state = self.integrator.step(self.state, self.model, self.environment)

        self.state = self.constraints.enforce_state_constraints(self.state)

    @abstractmethod
    def get_setpoints(self) -> Setpoints:
        """
        Generates the desired setpoints for the drone's operation.

        This method processes the current flight mode and control inputs to produce
        the desired setpoints, such as target positions, velocities, orientations, 
        or thrust levels, required for the drone's operation.

        Returns:
            Setpoints: An object containing the calculated setpoints.
        """
        pass

    

        
