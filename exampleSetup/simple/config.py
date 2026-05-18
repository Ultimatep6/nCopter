from __future__ import annotations

import numpy as np
from typing import List

from quad_sim.bases.configuration import BuildableConfig
from quad_sim.bases.rigidbody import RigidBody
from quad_sim.bases.state import StateVector
from quad_sim.references.bodyFixed import BodyFixed
from quad_sim.utils.decorators import topLevel

from classes import (
    DefaultAllocator,
    DefaultController,
    DefaultDynamics,
    DefaultEnvironment,
    DefaultIntegrator,
    DefaultMotor,
    DefaultPilot,
    DefaultDrone,
    DefaultConstraints,
    WindEffect,
    CirclePilot,
    CircleDrone,
)


# ── Motor ────────────────────────────────────────────────────────────────────


class MotorConfig(BuildableConfig):
    """Configuration for a single motor."""

    def __init__(
        self,
        motor_id: str="motor_1",
        spin_direction: int=1,
        position: list[float]=[0.0, 0.0, 0.0],
    ):
        super().__init__(DefaultMotor)
        self.motor_id = motor_id
        self.spin_direction = spin_direction
        self.position = position

    def construct(self) -> DefaultMotor:
        pos = self.position
        if isinstance(pos, list):
            pos = BodyFixed(pos[0], pos[1], pos[2])
        return DefaultMotor(
            id=self.motor_id,
            spin_direction=self.spin_direction,
            position=pos,
        )


# ── Dynamics ─────────────────────────────────────────────────────────────────


class DynamicsConfig(BuildableConfig):
    """Configuration for the drone dynamics (rigid body + motors)."""

    def __init__(
        self,
        mass: float = 1.0,
        inertia_tensor: np.ndarray = np.eye(3) * 0.02,
        motors: List[MotorConfig] = None,
    ):
        super().__init__(DefaultDynamics)
        self.mass = mass
        self.inertia_tensor = inertia_tensor
        if motors is None:
            arm = 0.15
            motors = [
                MotorConfig(motor_id="motor_0", spin_direction=1, position=[arm, 0.0, 0.0]),
                MotorConfig(motor_id="motor_1", spin_direction=-1, position=[0.0, -arm, 0.0]),
                MotorConfig(motor_id="motor_2", spin_direction=1, position=[-arm, 0.0, 0.0]),
                MotorConfig(motor_id="motor_3", spin_direction=-1, position=[0.0, arm, 0.0]),
            ]
        self.motors = motors

    def construct(self) -> DefaultDynamics:
        return DefaultDynamics(
            body=RigidBody(mass=self.mass, inertia_tensor=self.inertia_tensor),
            motors=[m.construct() for m in self.motors],
        )


# ── Pilot ────────────────────────────────────────────────────────────────────


class PilotConfig(BuildableConfig):
    """Configuration for the pilot / autopilot."""

    def __init__(self):
        super().__init__(DefaultPilot)

    def construct(self) -> DefaultPilot:
        return DefaultPilot()


# ── Allocator ────────────────────────────────────────────────────────────────


class AllocatorConfig(BuildableConfig):
    """Configuration for the control allocator."""

    def __init__(self, arm_length: float = 0.15, km_ratio: float = 0.05, kf: float = 1e-6):
        super().__init__(DefaultAllocator)
        self.arm_length = arm_length
        self.km_ratio = km_ratio
        self.kf = kf

    def construct(self) -> DefaultAllocator:
        return DefaultAllocator(
            arm_length=self.arm_length,
            km_ratio=self.km_ratio,
            kf=self.kf,
        )


# ── Controller ───────────────────────────────────────────────────────────────


class ControllerConfig(BuildableConfig):
    """Configuration for the radio / input controller."""

    def __init__(self):
        super().__init__(DefaultController)

    def construct(self) -> DefaultController:
        return DefaultController()


# ── Integrator ───────────────────────────────────────────────────────────────


class IntegratorConfig(BuildableConfig):
    """Configuration for the numerical integrator."""

    def __init__(self, dt: float=0.005):
        super().__init__(DefaultIntegrator)
        self.dt = dt

    def construct(self) -> DefaultIntegrator:
        return DefaultIntegrator(dt=self.dt)


# ── Environment ──────────────────────────────────────────────────────────────


class EnvironmentConfig(BuildableConfig):
    """Configuration for the simulation environment and its effects."""

    def __init__(self, effects: list | None = None):
        super().__init__(DefaultEnvironment)
        self.effects = effects

    def construct(self) -> DefaultEnvironment:
        if self.effects is None:
            return DefaultEnvironment(effects=[])
        return DefaultEnvironment(effects=self.effects)


# ── Constraints ──────────────────────────────────────────────────────────────

from quad_sim.bases.constraint import Constraint

class ConstraintConfig(BuildableConfig):
    """Configuration for the state constraints applied after each integration step."""

    def __init__(self, constraints: list[Constraint] | None = None):
        super().__init__(DefaultConstraints)
        self.constraints = constraints

    def construct(self) -> DefaultConstraints:
        if self.constraints is None:
            return DefaultConstraints()  # ships with sensible defaults
        return DefaultConstraints(constraints=self.constraints)


# ── Drone ────────────────────────────────────────────────────────────────────

@topLevel()
class DroneConfig(BuildableConfig):
    """
    Top-level configuration that composes every subsystem config
    and constructs a fully-assembled DefaultDrone.
    """

    def __init__(
        self,
        drone_id: str,
        init_state: StateVector | None = StateVector(),
        pilot: PilotConfig = PilotConfig(),
        dynamics: DynamicsConfig = DynamicsConfig(),
        allocator: AllocatorConfig = AllocatorConfig(),
        controller: ControllerConfig = ControllerConfig(),
        integrator: IntegratorConfig = IntegratorConfig(dt=0.005),
        environment: EnvironmentConfig = EnvironmentConfig(),
        constraints: ConstraintConfig = ConstraintConfig(),
    ):
        super().__init__(DefaultDrone)
        self.drone_id = drone_id
        self.init_state = init_state
        self.pilot = pilot
        self.dynamics = dynamics
        self.allocator = allocator
        self.controller = controller
        self.integrator = integrator
        self.environment = environment
        self.constraints = constraints

    def construct(self) -> DefaultDrone:
        if self.dynamics is None:
            raise ValueError(
                "DynamicsConfig must be provided before constructing a drone."
            )

        return DefaultDrone(
            drone_id=self.drone_id,
            init_state=self.init_state if self.init_state is not None else StateVector(),
            pilot=self.pilot.construct(),
            dynamics=self.dynamics.construct(),
            allocator=self.allocator.construct(),
            controller=self.controller.construct(),
            integrator=self.integrator.construct(),
            environment=self.environment.construct(),
            constraints=self.constraints.construct(),
        )


# ── Circle Pilot ─────────────────────────────────────────────────────────────


class CirclePilotConfig(BuildableConfig):
    """Configuration for a pilot that flies the drone in a horizontal circle."""

    def __init__(
        self,
        radius: float = 1.0,
        altitude: float = 2.0,
        angular_speed: float = 1.0,
        dt: float = 0.005,
        mass: float = 1.0,
    ):
        super().__init__(CirclePilot)
        self.radius = radius
        self.altitude = altitude
        self.angular_speed = angular_speed
        self.dt = dt
        self.mass = mass

    def construct(self) -> CirclePilot:
        return CirclePilot(
            radius=self.radius,
            altitude=self.altitude,
            angular_speed=self.angular_speed,
            dt=self.dt,
            mass=self.mass,
        )


# ── Circle Drone ─────────────────────────────────────────────────────────────


@topLevel()
class CircleDroneConfig(BuildableConfig):
    """Top-level drone config that uses CirclePilot and constructs a CircleDrone."""

    def __init__(
        self,
        drone_id: str,
        init_state: StateVector | None = None,
        pilot: CirclePilotConfig = CirclePilotConfig(),
        dynamics: DynamicsConfig = DynamicsConfig(),
        allocator: AllocatorConfig = AllocatorConfig(),
        controller: ControllerConfig = ControllerConfig(),
        integrator: IntegratorConfig = IntegratorConfig(dt=0.005),
        environment: EnvironmentConfig = EnvironmentConfig(),
        constraints: ConstraintConfig = ConstraintConfig(),
    ):
        super().__init__(CircleDrone)
        self.drone_id = drone_id
        if init_state is None:
            from quad_sim.references.earthFixed import EarthFixed
            from quad_sim.references.bodyFixed import BodyFixed
            from quad_sim.orientation.quaternion import Quaternion
            init_state = StateVector(
                position=EarthFixed(0.0, 0.0, 0.0),
                velocity=BodyFixed(0.0, 0.0, 0.0),
                quaternion=Quaternion(1.0, 0.0, 0.0, 0.0),
            )
        self.init_state = init_state
        self.pilot = pilot
        self.dynamics = dynamics
        self.allocator = allocator
        self.controller = controller
        self.integrator = integrator
        self.environment = environment
        self.constraints = constraints

    def construct(self) -> CircleDrone:
        if self.dynamics is None:
            raise ValueError(
                "DynamicsConfig must be provided before constructing a drone."
            )

        return CircleDrone(
            drone_id=self.drone_id,
            init_state=self.init_state if self.init_state is not None else StateVector(),
            pilot=self.pilot.construct(),
            dynamics=self.dynamics.construct(),
            allocator=self.allocator.construct(),
            controller=self.controller.construct(),
            integrator=self.integrator.construct(),
            environment=self.environment.construct(),
            constraints=self.constraints.construct(),
        )
