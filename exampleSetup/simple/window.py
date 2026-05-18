import moderngl
import numpy as np
import moderngl_window as mglw
import os
import sys

# Add parent dirs so quad_sim and vizcore can be found
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, "/home/matteobenk/myStuff/codingStuff/VizCore")
sys.path.insert(0, "/home/matteobenk/myStuff/codingStuff/VizCore/test_folder")

from vizcore.renderer.camera import Camera
from mesh_loader import load_model, create_example_cube
from vizcore.utils import get_total_rot_matrix, get_model_matrix

from config import CircleDroneConfig, CirclePilotConfig
from quad_sim.bases.sim import NCopterBase

ROOM_SIZE = 10.0
MIN_SIZE = 1.0
WALL_MARGIN = 0.2

AXIS_RED = np.array([0.9, 0.2, 0.2], dtype=np.float32)
AXIS_GREEN = np.array([0.2, 0.85, 0.2], dtype=np.float32)
AXIS_BLUE = np.array([0.2, 0.4, 0.95], dtype=np.float32)
INDICATOR_COLOR = np.array([1.0, 0.5, 0.15], dtype=np.float32)
MESH_COLOR = np.array([0.3, 0.8, 0.9], dtype=np.float32)
DRONE_COLOR = np.array([1.0, 0.8, 0.2], dtype=np.float32)

CIRCLE_SEGMENTS = 32
CIRCLE_RADIUS_FACTOR = 0.02
LINE_RADIUS_FACTOR = 0.003
AXIS_SCALE_FACTOR = 1.5
AXIS_DIVISOR = 10.0

INDICATOR_RADIUS = 0.3

MOVING_AXIS_X_AMP = 3.0
MOVING_AXIS_Y_AMP = 2.0
MOVING_AXIS_Y_OFFSET = 1.0
MOVING_AXIS_Y_FREQ = 0.7
MOVING_AXIS_Z_AMP = 3.0
MOVING_AXIS_ROT_SPEED = 1.2
MESH_SCALE_FACTOR = 0.001

FOG_DENSITY = 0.025
GRID_LINE_ALPHA = 1.0
GRID_ZONE_ALPHA = 0.9

SHADER_DIR = os.path.join(
    "/home/matteobenk/myStuff/codingStuff/VizCore/test_folder", "shaders"
)


class SimulationRoom(mglw.WindowConfig):
    """Simulation room with dynamic bounding box and 4 corner cameras.

    Features:
    - 5-face grid (floor + 4 walls) with LOD-based detail
    - 8 cameras (4 upper, 4 lower corners), switchable with A/D and W/S
    - Origin axis at room center
    - Dynamic second axis that moves around with height indicator
    - Live drone simulation rendered as a mesh
    """

    gl_version = (3, 3)
    window_size = (1024, 768)
    title = "Simulation Room"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        self.room_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.room_size = ROOM_SIZE
        self.min_size = MIN_SIZE
        self.current_camera_index = 0

        self._load_all_shaders()
        self._setup_grid_program()
        self._setup_cameras()
        self._setup_axis()
        self._setup_drone_simulation()

    def _load_all_shaders(self):
        # ----------------------- GRID PROGRAM
        self.grid_program = self.load_program(
            vertex_shader=os.path.join(SHADER_DIR, "sim_grid.vert"),
            fragment_shader=os.path.join(SHADER_DIR, "sim_grid.frag"),
        )

        # ----------------------- AXIS PROGRAM
        self.axis_program = self.load_program(
            vertex_shader=os.path.join(SHADER_DIR, "sim_axis.vert"),
            fragment_shader=os.path.join(SHADER_DIR, "sim_axis.frag"),
        )

        # ----------------------- MODEL PROGRAM
        self.model_program = self.load_program(
            vertex_shader=os.path.join(SHADER_DIR, "sim_model.vert"),
            fragment_shader=os.path.join(SHADER_DIR, "sim_model.frag"),
        )

        # ----------------------- INDICATOR PROGAM
        self.indicator_program = self.load_program(
            vertex_shader=os.path.join(SHADER_DIR, "sim_circle_indicator.vert"),
            fragment_shader=os.path.join(SHADER_DIR, "sim_circle_indicator.frag"),
        )

        # ------------------------ CYLINDER BEAM PROGRAM
        self.cylinder_program = self.load_program(
            vertex_shader=os.path.join(SHADER_DIR, "sim_cylinder_beam.vert"),
            fragment_shader=os.path.join(SHADER_DIR, "sim_cylinder_beam.frag"),
        )

    def _setup_drone_simulation(self):
        self.sim = NCopterBase(
            agents=[
                CircleDroneConfig(
                    drone_id="drone_1",
                    pilot=CirclePilotConfig(
                        radius=0.0, altitude=2.0, angular_speed=0.0, dt=0.005
                    ),
                )
            ]
        )
        self.sim_steps = 0
        self.steps_per_frame = 5
        self._alt_timer = 0.0
        self._alt_targets = [1.5, 2.5, 3.5, 2.0]
        self._alt_index = 0

        # Load mesh for drone visualization
        mesh_path = os.path.join(os.path.dirname(__file__), "droneMod.obj")
        if os.path.exists(mesh_path):
            mesh_data = load_model(mesh_path)
            print(f"Loaded drone model from {mesh_path}")
        else:
            mesh_data = create_example_cube()
            print(f"Drone model not found, using unit cube fallback")

        self.vbo_drone_vert = self.ctx.buffer(mesh_data["vertices"].astype("f4"))
        self.vbo_drone_normal = self.ctx.buffer(mesh_data["normals"].astype("f4"))
        self.ibo_drone = self.ctx.buffer(mesh_data["indices"])

        self.drone_vao = self.ctx.vertex_array(
            self.model_program,
            [
                (self.vbo_drone_vert, "3f", "in_vert"),
                (self.vbo_drone_normal, "3f", "in_normal"),
            ],
            index_buffer=self.ibo_drone,
        )
        self.drone_index_count = len(mesh_data["indices"]) * 3
        print(
            f"Drone mesh: {len(mesh_data['vertices'])} vertices, {self.drone_index_count} indices"
        )

    def _setup_grid_program(self):

        self._setup_grid_geometry()

    def _setup_grid_geometry(self, wall_direction=1):
        half_size = self.room_size * 0.5
        wall_top = self.room_size * wall_direction

        floor = np.array(
            [
                [-half_size, 0.0, -half_size],
                [half_size, 0.0, -half_size],
                [half_size, 0.0, half_size],
                [-half_size, 0.0, -half_size],
                [half_size, 0.0, half_size],
                [-half_size, 0.0, half_size],
            ],
            dtype=np.float32,
        )

        wall_pos_x = np.array(
            [
                [half_size, 0.0, -half_size],
                [half_size, wall_top, -half_size],
                [half_size, wall_top, half_size],
                [half_size, 0.0, -half_size],
                [half_size, wall_top, half_size],
                [half_size, 0.0, half_size],
            ],
            dtype=np.float32,
        )

        wall_neg_x = np.array(
            [
                [-half_size, 0.0, half_size],
                [-half_size, wall_top, half_size],
                [-half_size, wall_top, -half_size],
                [-half_size, 0.0, half_size],
                [-half_size, wall_top, -half_size],
                [-half_size, 0.0, -half_size],
            ],
            dtype=np.float32,
        )

        wall_pos_z = np.array(
            [
                [-half_size, 0.0, half_size],
                [half_size, 0.0, half_size],
                [half_size, wall_top, half_size],
                [-half_size, 0.0, half_size],
                [half_size, wall_top, half_size],
                [-half_size, wall_top, half_size],
            ],
            dtype=np.float32,
        )

        wall_neg_z = np.array(
            [
                [half_size, 0.0, -half_size],
                [-half_size, 0.0, -half_size],
                [-half_size, wall_top, -half_size],
                [half_size, 0.0, -half_size],
                [-half_size, wall_top, -half_size],
                [half_size, wall_top, -half_size],
            ],
            dtype=np.float32,
        )

        vertices = np.concatenate(
            [floor, wall_pos_x, wall_neg_x, wall_pos_z, wall_neg_z], axis=0
        )

        self.vbo_grid = self.ctx.buffer(vertices.astype("f4"))
        self.grid_vao = self.ctx.vertex_array(
            self.grid_program,
            [(self.vbo_grid, "3f", "in_position")],
        )

    def _setup_cameras(self):
        self.cameras = []
        for _ in range(8):
            camera = Camera(
                window_size=self.window_size,
                eye=np.array([0.0, 0.0, 0.0]),
                move_speed=0.0,
            )
            self.cameras.append(camera)

        self.grid_camera = Camera(
            window_size=self.window_size,
            eye=np.array([0.0, 0.0, 0.0]),
            target=np.array([0.0, 0.0, 0.0]),
            move_speed=0.0,
        )

        self._update_camera_positions()

    def _update_camera_positions(self):
        half_size = self.room_size * 0.5

        upper_positions = [
            np.array([-half_size, self.room_size, -half_size], dtype=np.float32),
            np.array([half_size, self.room_size, -half_size], dtype=np.float32),
            np.array([-half_size, self.room_size, half_size], dtype=np.float32),
            np.array([half_size, self.room_size, half_size], dtype=np.float32),
        ]

        lower_positions = [
            np.array([-half_size, -self.room_size, -half_size], dtype=np.float32),
            np.array([half_size, -self.room_size, -half_size], dtype=np.float32),
            np.array([-half_size, -self.room_size, half_size], dtype=np.float32),
            np.array([half_size, -self.room_size, half_size], dtype=np.float32),
        ]

        for i in range(4):
            self.cameras[i].eye = upper_positions[i] + self.room_center
            self.cameras[i].target = np.array(self.room_center, dtype=np.float32)

        for i in range(4):
            self.cameras[i + 4].eye = lower_positions[i] + self.room_center
            self.cameras[i + 4].target = np.array(self.room_center, dtype=np.float32)

        self.grid_camera.eye = upper_positions[0] + self.room_center
        self.grid_camera.target = np.array(self.room_center, dtype=np.float32)

    def _setup_axis(self):

        from vizcore.meshes.arrow import build_axis_arrow
        from vizcore.utils import get_rot_matrix, X_AXIS, Y_AXIS

        arrow_verts, arrow_idxs = build_axis_arrow(
            shaft_radius=0.02,
            shaft_height=0.5,
            head_radius=0.06,
            head_height=0.1,
            segments=16,
        )

        def rotate_arrow(verts, axis):
            if axis == "x":
                return verts @ get_rot_matrix(np.pi / 2, Y_AXIS, dim=True).T
            elif axis == "y":
                return verts @ get_rot_matrix(-np.pi / 2, X_AXIS, dim=True).T
            else:
                return verts

        x_verts = rotate_arrow(arrow_verts, "x")
        y_verts = rotate_arrow(arrow_verts, "y")
        z_verts = rotate_arrow(arrow_verts, "z")

        n_x = len(x_verts)
        n_y = len(y_verts)
        n_z = len(z_verts)

        all_verts = np.concatenate([x_verts, y_verts, z_verts], axis=0)
        all_colors = np.concatenate(
            [
                np.tile(AXIS_RED, (n_x, 1)),
                np.tile(AXIS_GREEN, (n_y, 1)),
                np.tile(AXIS_BLUE, (n_z, 1)),
            ],
            axis=0,
        )

        all_idxs = np.concatenate(
            [
                arrow_idxs,
                arrow_idxs + n_x,
                arrow_idxs + n_x + n_y,
            ],
            axis=0,
        ).astype(np.int32)

        self.vbo_axis_vert = self.ctx.buffer(all_verts[:, :3].astype("f4"))
        self.vbo_axis_color = self.ctx.buffer(all_colors.astype("f4"))
        self.ibo_axis = self.ctx.buffer(all_idxs)

        self.axis_vao = self.ctx.vertex_array(
            self.axis_program,
            [
                (self.vbo_axis_vert, "3f", "in_vert"),
                (self.vbo_axis_color, "3f", "in_color"),
            ],
            index_buffer=self.ibo_axis,
        )

        self._build_circle_geometry()
        self._build_cylinder_geometry()

    def _build_circle_geometry(self):
        # Indicator shader uses gl_VertexID only, no vertex attributes needed.
        # moderngl requires at least an empty VAO to render.
        self.circle_vao = self.ctx.vertex_array(
            self.indicator_program,
            [],
        )

    def _build_cylinder_geometry(self):
        self.cylinder_vao = self.ctx.vertex_array(self.cylinder_program, [])
        self.max_cylinders = 16
        self.cylinder_instance_matrices = np.zeros(
            (self.max_cylinders, 4, 4), dtype=np.float32
        )
        self.cylinder_instance_colors = np.zeros(
            (self.max_cylinders, 3), dtype=np.float32
        )
        self.cylinder_vert_count = 24 * 6

    def compute_room_bounds(self, assets: list):
        if not assets:
            return

        assets_arr = np.array(assets, dtype=np.float32)
        asset_mins = np.min(assets_arr, axis=0)
        asset_maxs = np.max(assets_arr, axis=0)

        half_size = self.room_size * 0.5
        room_min = self.room_center - half_size
        room_max = self.room_center + half_size

        new_min = np.minimum(room_min, asset_mins)
        new_max = np.maximum(room_max, asset_maxs)

        margin = (new_max - new_min) * WALL_MARGIN
        new_min -= margin
        new_max += margin

        new_center = (new_min + new_max) / 2.0
        new_size = np.max(new_max - new_min)
        new_size = max(new_size, self.min_size)

        self.room_center = new_center
        self.room_size = new_size

        self._update_camera_positions()
        self._setup_grid_geometry()

    def _get_active_camera(self):
        return self.cameras[self.current_camera_index]

    def on_key_event(self, key, action, modifiers):
        if action == self.wnd.keys.ACTION_PRESS:
            if key == self.wnd.keys.A:
                base = 0 if self.current_camera_index < 4 else 4
                self.current_camera_index = (
                    base + (self.current_camera_index - base - 1) % 4
                )
            elif key == self.wnd.keys.D:
                base = 0 if self.current_camera_index < 4 else 4
                self.current_camera_index = (
                    base + (self.current_camera_index - base + 1) % 4
                )
            elif key == self.wnd.keys.W:
                if self.current_camera_index >= 4:
                    self.current_camera_index -= 4
                    self._setup_grid_geometry(1)
            elif key == self.wnd.keys.S:
                if self.current_camera_index < 4:
                    self.current_camera_index += 4
                    self._setup_grid_geometry(-1)

    def _step_simulation(self):
        for _ in range(self.steps_per_frame):
            self.sim.run()
            self.sim_steps += 1

    def _get_drone_state(self):
        drone = self.sim.entities["drone_1"]
        pos = drone.state.position.vec
        q = drone.state.quaternion
        return pos, q

    def _quaternion_to_rotation_matrix(self, q):
        w, x, y, z = q.w, q.x, q.y, q.z
        R_sim = np.array(
            [
                [
                    1 - 2 * y * y - 2 * z * z,
                    2 * x * y - 2 * w * z,
                    2 * x * z + 2 * w * y,
                ],
                [
                    2 * x * y + 2 * w * z,
                    1 - 2 * x * x - 2 * z * z,
                    2 * y * z - 2 * w * x,
                ],
                [
                    2 * x * z - 2 * w * y,
                    2 * y * z + 2 * w * x,
                    1 - 2 * x * x - 2 * y * y,
                ],
            ],
            dtype=np.float32,
        )
        R_mesh_to_sim = np.array(
            [
                [1, 0, 0],
                [0, 0, 1],
                [0, -1, 0],
            ],
            dtype=np.float32,
        )
        T = np.array(
            [
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0],
            ],
            dtype=np.float32,
        )
        R_vis = T @ R_sim @ R_mesh_to_sim
        return R_vis

    def on_render(self, time, frame_time):
        self._step_simulation()
        drone_pos, drone_q = self._get_drone_state()

        # Drone position from simulation (sim z-down → vis y-up)
        moving_offset = np.array(
            [drone_pos[0, 0], -drone_pos[2, 0], drone_pos[1, 0]],
            dtype=np.float32,
        )

        if self.sim_steps % 50 == 0:
            euler = drone_q.to_euler()
            print(
                f"Sim step {self.sim_steps} | pos: ({moving_offset[0]:.2f}, {moving_offset[1]:.2f}, {moving_offset[2]:.2f}) | yaw={np.degrees(euler.yaw):.1f}° pitch={np.degrees(euler.pitch):.1f}° roll={np.degrees(euler.roll):.1f}°"
            )

        camera = self._get_active_camera()
        vp = camera.get_vp()

        self._alt_timer += frame_time
        if self._alt_timer >= 3.0:
            self._alt_timer = 0.0
            self._alt_index = (self._alt_index + 1) % len(self._alt_targets)
            self.sim.entities["drone_1"].pilot.altitude = self._alt_targets[
                self._alt_index
            ]

        # Clear the window
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.ctx.disable(moderngl.DEPTH_TEST)
        # ---------------------- Grid Program Setup

        self.grid_program["gVp"].write(vp.astype("f4").T.tobytes())
        self.grid_program["gRoomCenter"].write(self.room_center.astype("f4").tobytes())
        self.grid_program["gRoomSize"].write(
            np.array([self.room_size], dtype="f4").tobytes()
        )
        self.grid_program["gCameraPos"].write(camera.eye.astype("f4").tobytes())
        self.grid_program["gFogDensity"].write(
            np.array([FOG_DENSITY], dtype="f4").tobytes()
        )
        self.grid_program["gGridLineAlpha"].write(
            np.array([GRID_LINE_ALPHA], dtype="f4").tobytes()
        )
        self.grid_program["gGridZoneAlpha"].write(
            np.array([GRID_ZONE_ALPHA], dtype="f4").tobytes()
        )

        self.grid_vao.render(moderngl.TRIANGLES, 30)
        self.ctx.enable(moderngl.DEPTH_TEST)

        # INDICATOR SETUP
        px = float(drone_pos[0, 0])
        pz = float(drone_pos[1, 0])
        py = float(drone_pos[2, 0])
        mvp_center = get_model_matrix(tx=px, ty=0, tz=pz)
        mvp_center = vp @ mvp_center
        pilot = self.sim.entities["drone_1"].pilot
        R = INDICATOR_RADIUS
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.indicator_program["mvp"].write(mvp_center.astype("f4").T.tobytes())
        self.indicator_program["R"].value = R
        self.indicator_program["u_color"].write(INDICATOR_COLOR.astype("f4").tobytes())
        self.circle_vao.render(moderngl.TRIANGLE_STRIP, vertices=4)

        # ------------------------------ CYLINDER BEAM SETUP
        self.cylinder_instance_matrices[:] = 0.0
        self.cylinder_instance_colors[:] = 0.0
        cyl_radius = INDICATOR_RADIUS / 4.0
        self.cylinder_instance_matrices[0] = get_model_matrix(
            tx=px, ty=0.0, tz=pz, sx=cyl_radius, sy=-py, sz=cyl_radius
        )
        self.cylinder_instance_colors[0] = INDICATOR_COLOR
        self.cylinder_program["u_vp"].write(vp.astype("f4").T.tobytes())
        self.cylinder_program["u_models"].write(
            self.cylinder_instance_matrices.transpose(0, 2, 1)
            .ravel()
            .astype("f4")
            .tobytes()
        )
        self.cylinder_program["u_colors"].write(
            self.cylinder_instance_colors.astype("f4").tobytes()
        )
        self.cylinder_vao.render(
            moderngl.TRIANGLES, self.cylinder_vert_count, instances=1
        )

        # ------------------------------ AXIS SETUP
        axis_scale = (self.room_size / AXIS_DIVISOR) * AXIS_SCALE_FACTOR
        self.axis_program["u_scale"].write(np.array([axis_scale], dtype="f4").tobytes())
        self.axis_program["u_vp"].write(vp.astype("f4").T.tobytes())

        identity = np.eye(3, dtype=np.float32)
        self.axis_program["u_rotation"].write(identity.T.tobytes())

        self.axis_program["u_offset"].write(
            np.array([0.0, 0.0, 0.0], dtype=np.float32).tobytes()
        )
        self.axis_program["u_use_override"].write(np.array([0], dtype="i4").tobytes())
        self.axis_vao.render(moderngl.TRIANGLES)

        # ---------------------------- MODEL SETUP
        rotation = self._quaternion_to_rotation_matrix(drone_q)

        self.model_program["u_vp"].write(vp.astype("f4").T.tobytes())
        self.model_program["u_offset"].write(moving_offset.tobytes())
        self.model_program["u_scale"].write(
            np.array([MESH_SCALE_FACTOR], dtype="f4").tobytes()
        )
        self.model_program["u_rotation"].write(rotation.T.tobytes())
        self.model_program["u_color"].write(DRONE_COLOR.tobytes())
        self.drone_vao.render(moderngl.TRIANGLES, self.drone_index_count)

        self.ctx.enable(moderngl.DEPTH_TEST)


if __name__ == "__main__":
    SimulationRoom.run()
