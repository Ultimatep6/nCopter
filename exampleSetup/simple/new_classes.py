import numpy as np
from numpy.linalg import norm

# Patch nCopter funcs.py runtime bug: types only imported under TYPE_CHECKING
import quad_sim.funcs as _funcs
from quad_sim.references.bodyFixed import BodyFixed as _BodyFixed
from quad_sim.references.earthFixed import EarthFixed as _EarthFixed
from quad_sim.orientation.quaternion import Quaternion as _Quaternion

_funcs.BodyFixed = _BodyFixed
_funcs.EarthFixed = _EarthFixed
_funcs.Quaternion = _Quaternion

from quad_sim.bases.allocator import AllocatorBase, BodyFixed, Tuple
from quad_sim.bases.controller import ControllerBase, List
from quad_sim.bases.dynamics import DynamicsBase, RigidBody, MotorBase
from quad_sim.bases.environment import EnvironmentBase
from quad_sim.bases.pilot import PilotBase
from quad_sim.bases.pid import PIDBase
from quad_sim.bases.drone import DroneBase, StateVector, Setpoints
from quad_sim.bases.integrator import IntegratorBase
from quad_sim.bases.environment import EnvironmentBase, EnvironmentEffect
from quad_sim.bases.constraint import (
    StateConstraint,
    SetpointConstraint,
    ConstraintBase,
)

from quad_sim.orientation.quaternion import Quaternion
from quad_sim.references.earthFixed import EarthFixed
from quad_sim.funcs import _get_body_to_inertial


# ── Allocator ─────────────────────────────────────────────────────────────
class DefaultAllocator(AllocatorBase):
    """
    Mixer for + configuration quadcopter.

    Motor layout (NED body frame, X-forward, Y-left):
      Motor 0: front  (x=+l, y=0),  CW  (+1)
      Motor 1: right  (x=0,  y=-l), CCW (-1)
      Motor 2: back   (x=-l, y=0),  CW  (+1)
      Motor 3: left   (x=0,  y=+l), CCW (-1)

    Mixer equations (inverted):
      ω²_0 = T/(4kf)
      ω²_1 = T/(4kf)
      ω²_2 = T/(4kf)
      ω²_3 = T/(4kf)
    """

    def __init__(
        self, arm_length: float = 0.15, km_ratio: float = 0.05, kf: float = 1e-6
    ):
        self.arm_length = arm_length
        self.km_ratio = km_ratio
        self.kf = kf

    def allocate(
        self, thrust_torques: Tuple[BodyFixed, BodyFixed], n_motors: int = 4
    ) -> list[float]:
        desired_thrust, desired_torque = thrust_torques

        T = desired_thrust.vec[2, 0]
        Mx = desired_torque.vec[0, 0]
        My = desired_torque.vec[1, 0]
        Mz = desired_torque.vec[2, 0]

        l = self.arm_length
        km = self.km_ratio * self.kf
        kf = self.kf

        thrust_per_motor = np.zeros(4)
        thrust_per_motor[0] = T / (4 * kf)
        thrust_per_motor[1] = T / (4 * kf)
        thrust_per_motor[2] = T / (4 * kf)
        thrust_per_motor[3] = T / (4 * kf)

        thrust_per_motor = np.clip(thrust_per_motor, 0, None)

        rpms = np.sqrt(thrust_per_motor)

        return rpms.tolist()


# ── Controller ─────────────────────────────────────────────────────────────
class DefaultController(ControllerBase):
    def __init__(self):
        self.__connected = (
            True  # Simulate a successful connection for demonstration purposes
        )

        self.channels = {
            "throttle": 0.0,
            "roll_angle": 0.0,
            "pitch_angle": 0.0,
            "yaw_angle": 0.0,
        }
        self.switches = {"mode_switch": False}

    def connect(self) -> bool:
        # Placeholder implementation: Simulate a successful connection // Always return True for demonstration purposes
        return True

    def calibrate(
        self, min: int | dict, max: int | dict, trim: int | dict, offset: int | dict
    ) -> bool:
        # Placeholder implementation: Simulate successful calibration
        return True

    def get_axis_value(self, channel_id: str | List[str]) -> float | dict:
        if not all(
            isinstance(ch, str)
            for ch in (channel_id if isinstance(channel_id, list) else [channel_id])
        ):
            raise ValueError(
                "channel_id must be a string or a list of strings representing channel names."
            )
        if isinstance(channel_id, str):
            return self.channels.get(channel_id, 0.0)
        elif isinstance(channel_id, list):
            return {ch: self.channels.get(ch, 0.0) for ch in channel_id}
        return 0.0

    def get_switch_value(self, switch_id: str | List[str]) -> float | dict:
        if not all(
            isinstance(sw, str)
            for sw in (switch_id if isinstance(switch_id, list) else [switch_id])
        ):
            raise ValueError(
                "switch_id must be a string or a list of strings representing switch names."
            )
        if isinstance(switch_id, str):
            return self.switches.get(switch_id, False)
        elif isinstance(switch_id, list):
            return {sw: self.switches.get(sw, False) for sw in switch_id}
        return 0.0

    def is_connected(self) -> bool:
        # Placeholder implementation: Simulate that the controller is always connected
        return self.__connected is True


# ── Motors ─────────────────────────────────────────────────────────────
class DefaultMotor(MotorBase):
    def __init__(
        self,
        id: str,
        spin_direction: int,
        position: BodyFixed,
        kf: float = 1e-6,
        km: float = 1e-7,
        propLength: float = 0.1,
        nProps: int = 2,
        motor_tau: float = 0.005,
    ):
        super().__init__(id, spin_direction, position)

        self.rpm = 0.0
        self.rpm_cmd = 0.0
        self.theta = 0.0
        self.motor_tau = motor_tau

        self.kf = kf
        self.km = km

        self.propTips = self._generate_propeller_tips(propLength, nProps)

    @property
    def iD(self) -> str:
        return self._iD

    @property
    def spin_direction(self) -> int:
        return self._spin_direction

    @property
    def position(self) -> BodyFixed:
        return self._position

    def compute_forces(self) -> Tuple[BodyFixed, BodyFixed]:
        thrust = BodyFixed(0.0, 0.0, -self.kf * self.rpm**2)
        torque = BodyFixed(0.0, 0.0, -self._spin_direction * self.km * self.rpm**2)
        return thrust, torque

    def set_rpm(self, rpm: float) -> None:
        self.rpm = max(rpm, 0.0)
        self.rpm_cmd = self.rpm

    def update_rpm(self, dt: float) -> None:
        pass

    def _generate_propeller_tips(
        self, propLength: float, nProps: int
    ) -> dict[str, BodyFixed]:
        propTips = {}
        for i in range(nProps):
            angle = (2 * np.pi / nProps) * i
            tip_x = self.position.x + propLength * np.cos(angle)
            tip_y = self.position.y + propLength * np.sin(angle)
            tip_z = self.position.z
            propTips[f"prop_{i}"] = BodyFixed(tip_x[0], tip_y[0], tip_z[0])
        return propTips

    def locate_propeller_tips(self):
        return self._generate_propeller_tips()

    def update_theta(self, dt):
        return self.theta + (self.rpm / 60.0) * 2 * np.pi * dt


# ── Dynamics ─────────────────────────────────────────────────────────────
class DefaultDynamics(DynamicsBase):
    def __init__(self, body: RigidBody, motors: List[MotorBase]):
        super().__init__(body, motors)

    @property
    def rotor_rates(self) -> dict[str, float]:
        return {f"Motor_{i}": motor.rpm for i, motor in enumerate(self.motors)}

    def compute_accelerations(self, F, M, state):
        from quad_sim.funcs import compute_aB, compute_alphaB

        a = compute_aB(self.mass, F, state.omega, state.velocity)
        alpha = compute_alphaB(self.inertia_tensor, M, state.omega)
        return a, alpha

    def set_motor_rpm(self, rpms: list[float]) -> None:
        for motor, rpm in zip(self.motors, rpms):
            motor.set_rpm(rpm)

    def update_motors(self, dt: float) -> None:
        for motor in self.motors:
            if hasattr(motor, "update_rpm"):
                motor.update_rpm(dt)


# ── Pilot ─────────────────────────────────────────────────────────────
class DefaultPilot(PilotBase):
    def compute_control(
        self, state: StateVector, target: Setpoints
    ) -> Tuple[BodyFixed, BodyFixed]:
        px = state.position.vec[0, 0]
        py = state.position.vec[1, 0]
        pz = state.position.vec[2, 0]

        vx = state.velocity.vec[0, 0]
        vy = state.velocity.vec[1, 0]
        vz = state.velocity.vec[2, 0]

        target_z = target.z if target.z is not None else 2.0

        Kp = 3.0
        Kd = 2.0
        g = 9.81

        uz = Kp * (target_z - pz) - Kd * vz + g

        thrust = BodyFixed(0.0, 0.0, uz)
        torque = BodyFixed(0.0, 0.0, 0.0)

        return thrust, torque


class CirclePilot(PilotBase):
    """
    Simple cascaded trajectory-tracking pilot.

    Architecture:
      Trajectory → Position → Velocity → Acceleration → Attitude → Torque

    Uses velocity/acceleration feedforward for smooth tracking.
    Yaw is held fixed to avoid cross-axis coupling.
    """

    def __init__(
        self, radius: float = 1.0, altitude: float = 2.0, angular_speed: float = 1.0
    ):
        self.radius = radius
        self.altitude = altitude
        self.angular_speed = angular_speed
        self._time = 0.0

        self._previous_error = 0.0
        self._integral = 0

        # PID Loop
        PID = PIDBase(
            kp=2.5,
            ki=0.0,
            kd=1.5,
        )

    def _trajectory(self, t):
        alpha = self.angular_speed
        pos_x = self.radius * np.cos(alpha * t)
        pos_y = self.radius * np.sin(alpha * t)
        pos_z = -self.altitude

        vel_x = -self.radius * alpha * np.sin(alpha * t)
        vel_y = self.radius * alpha * np.cos(alpha * t)
        vel_z = 0.0

        acc_x = -self.radius * alpha**2 * np.cos(alpha * t)
        acc_y = -self.radius * alpha**2 * np.sin(alpha * t)
        acc_z = 0.0

        return (pos_x, pos_y, pos_z), (vel_x, vel_y, vel_z), (acc_x, acc_y, acc_z)

    def compute_control(
        self, state: StateVector, target: Setpoints
    ) -> Tuple[BodyFixed, BodyFixed]:
        self._time += 0.01
        t = self._time

        px = state.position.vec[0, 0]
        py = state.position.vec[1, 0]
        pz = state.position.vec[2, 0]

        R = _get_body_to_inertial(state.quaternion.as_np())
        vel_earth = R @ state.velocity.vec
        vx = vel_earth[0, 0]
        vy = vel_earth[1, 0]
        vz = vel_earth[2, 0]

        (tx, ty, tz), (tvx, tvy, tvz), (tax, tay, taz) = self._trajectory(t)

        g = 9.81

        a_cmd = (
            np.array([tax, tay, taz])
            + PID._kp * PID.error(np.array([tx, ty, tz]), np.array([px, py, pz]))
            + PID._kd * PID.error(np.array([tvx, tvy, tvz]), np.array([pvx, pvy, pvz]))
        )

        a_cmd += np.array([0, 0, g])
        b_cmd = a_cmd / norm(a_cmd)

        # Desired roll and pitch from acceleration commands (small-angle approx)
        des_roll = np.clip(-des_ay / max(thrust_mag, 5.0), -0.3, 0.3)
        des_pitch = np.clip(des_ax / max(thrust_mag, 5.0), -0.3, 0.3)

        euler = state.quaternion.to_euler()
        desired_yaw = 0.0

        # ── PD attitude control ──
        Kp_att = 0.05
        Kd_att = 0.02

        roll_err = np.clip(des_roll - euler.roll, -0.5, 0.5)
        pitch_err = np.clip(des_pitch - euler.pitch, -0.5, 0.5)
        yaw_err = np.clip(desired_yaw - euler.yaw, -0.5, 0.5)

        torque_vec = np.array(
            [
                Kp_att * roll_err - Kd_att * state.omega.vec[0, 0],
                Kp_att * pitch_err - Kd_att * state.omega.vec[1, 0],
                Kp_att * 0.2 * yaw_err - Kd_att * state.omega.vec[2, 0],
            ],
            dtype=np.float32,
        )
        torque_vec = np.clip(torque_vec, -0.01, 0.01)

        thrust = BodyFixed(0.0, 0.0, thrust_mag)
        torque = BodyFixed(
            float(torque_vec[0]),
            float(torque_vec[1]),
            float(torque_vec[2]),
        )

        return thrust, torque


# ── Integrator ─────────────────────────────────────────────────────────────
class DefaultIntegrator(IntegratorBase):
    def __init__(self, dt: float):
        super().__init__(dt)

    def _compute_q_rate(self, quaternion, omega):
        p, q, r = omega.vec.ravel()
        w, x, y, z = quaternion.w, quaternion.x, quaternion.y, quaternion.z
        dw = 0.5 * (-p * x - q * y - r * z)
        dx = 0.5 * (p * w + r * y - q * z)
        dy = 0.5 * (q * w - r * x + p * z)
        dz = 0.5 * (r * w + q * x - p * y)
        return Quaternion(dw, dx, dy, dz)

    def integrate(
        self, acc: BodyFixed, alpha: BodyFixed, state: StateVector
    ) -> StateVector:
        q_rate = self._compute_q_rate(state.quaternion, state.omega)

        new_q = state.quaternion + q_rate * self.dt
        n = new_q.norm()
        if n > 1e-10:
            new_q = Quaternion(new_q.w / n, new_q.x / n, new_q.y / n, new_q.z / n)

        R = _get_body_to_inertial(new_q.as_np())

        gravity_earth = np.array([[0.0], [0.0], [9.81]], dtype=np.float32)
        gravity_body = R.T @ gravity_earth
        acc_total = BodyFixed.from_Array(acc.vec + gravity_body, flag="acceleration")

        vel_earth = R @ state.velocity.vec
        new_pos = EarthFixed.from_Array(
            state.position.vec + vel_earth * self.dt, flag="position"
        )

        new_vel = BodyFixed.from_Array(
            state.velocity.vec + acc_total.vec * self.dt, flag="velocity"
        )

        new_omega = BodyFixed.from_Array(
            state.omega.vec + alpha.vec * self.dt, flag="ang_velocity"
        )

        new_state = StateVector(
            position=new_pos,
            velocity=new_vel,
            quaternion=new_q,
            omega=new_omega,
            acceleration=acc_total,
            alpha=alpha,
        )
        return new_state

    def step(self, state0, model, environment):
        model.update_motors(self.dt)
        return super().step(state0, model, environment)


# ── Environment ─────────────────────────────────────────────────────────────
class WindEffect(EnvironmentEffect):
    def __init__(
        self, dir: BodyFixed = BodyFixed(1.0, 0.0, 0.0), magnitude: float = 0.0
    ):
        if np.linalg.norm(dir.vec) != 1:
            raise ValueError("Wind direction vector must be a unit vector.")

        self.force = dir * magnitude  # Wind force vector in the body-fixed frame

    def apply(self, state: StateVector) -> Tuple[BodyFixed, BodyFixed]:
        # Placeholder implementation: Return zero wind forces and moments
        return self.force, BodyFixed(0.0, 0.0, 0.0)


class DefaultEnvironment(EnvironmentBase):
    def __init__(self, effects: list[EnvironmentEffect] = [WindEffect()]):
        super().__init__(effects)


# ── Constraints ──────────────────────────────────────────────────────────────


class GroundPlaneConstraint(SetpointConstraint):
    """Prevents the drone from falling below a minimum altitude (z >= floor)."""

    def __init__(self, floor: float = 0.0):
        self.floor = floor

    def enforce(self, setpoint: Setpoints) -> StateVector:
        if setpoint.z < self.floor:
            setpoint.z = self.floor
        return setpoint


class MaxVelocityConstraint(StateConstraint):
    """Clamps the linear velocity magnitude to a maximum value."""

    def __init__(self, max_speed: float = 30.0):
        self.max_speed = max_speed

    def enforce(self, state: StateVector) -> StateVector:
        speed = float(np.linalg.norm(state.velocity.vec))
        if speed > self.max_speed:
            state.velocity.vec[:] = state.velocity.vec * (self.max_speed / speed)
        return state


class DefaultConstraints(ConstraintBase):
    """Ships with the standard set of constraints."""

    def __init__(
        self, constraints: list[StateConstraint | SetpointConstraint] | None = None
    ):
        if constraints is None:
            constraints = [
                GroundPlaneConstraint(),
                MaxVelocityConstraint(),
            ]
        super().__init__(constraints)


# ── Drone ────────────────────────────────────────────────────────────────────


class DefaultDrone(DroneBase):
    def __init__(
        self,
        drone_id,
        init_state,
        pilot,
        dynamics,
        allocator,
        controller,
        integrator,
        environment,
        constraints=None,
    ):
        super().__init__(
            drone_id,
            init_state,
            pilot,
            dynamics,
            allocator,
            controller,
            integrator,
            environment,
            constraints,
        )

    def get_setpoints(self):
        for iD in self.controller.channels:
            self.channels[iD] = self.controller.get_axis_value(iD)
        for iD in self.controller.switches:
            self.switches[iD] = self.controller.get_switch_value(iD)

        # Placeholder implementation: Return default setpoints based on controller input
        return Setpoints(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0)


# ── Circle Drone ─────────────────────────────────────────────────────────────


class CircleDrone(DroneBase):
    """Drone that emits circular trajectory setpoints for CirclePilot."""

    def __init__(
        self,
        drone_id,
        init_state,
        pilot,
        dynamics,
        allocator,
        controller,
        integrator,
        environment,
        constraints=None,
    ):
        super().__init__(
            drone_id,
            init_state,
            pilot,
            dynamics,
            allocator,
            controller,
            integrator,
            environment,
            constraints,
        )

    def get_setpoints(self):
        angle = self.pilot.angular_speed * self.pilot._time
        return Setpoints(
            x=self.pilot.radius * np.cos(angle),
            y=self.pilot.radius * np.sin(angle),
            z=self.pilot.altitude,
            roll=0.0,
            pitch=0.0,
            yaw=angle,
        )
