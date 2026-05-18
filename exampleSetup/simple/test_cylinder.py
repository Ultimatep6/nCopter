import moderngl
import moderngl_window as mglw
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, "/home/matteobenk/myStuff/codingStuff/VizCore")

from vizcore.utils import get_view_matrix, get_perspective_matrix, get_model_matrix

SHADER_DIR = "/home/matteobenk/myStuff/codingStuff/VizCore/test_folder/shaders"
INDICATOR_RADIUS = 0.5
MAX_INSTANCES = 16
CYLINDER_VERTS = 24 * 6


class CylinderTestWindow(mglw.WindowConfig):
    gl_version = (3, 3)
    window_size = (1280, 720)
    aspect_ratio = 1280 / 720
    title = "Instanced Cylinder Beam Test"
    resizable = True
    samples = 4

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ctx.enable(moderngl.DEPTH_TEST)

        self.eye = np.array([6.0, 4.0, 6.0], dtype=np.float32)
        self.target = np.array([0.0, 1.5, 0.0], dtype=np.float32)
        self.fov = 60.0
        self.near = 0.1
        self.far = 100.0
        self._yaw = -np.pi / 4
        self._pitch = -np.pi / 6
        self._dist = 8.0

        self.cylinder_program = self.load_program(
            vertex_shader=os.path.join(SHADER_DIR, "sim_cylinder_beam.vert"),
            fragment_shader=os.path.join(SHADER_DIR, "sim_cylinder_beam.frag"),
        )

        self.cylinder_vao = self.ctx.vertex_array(self.cylinder_program, [])

        self._build_demo_instances()

        self.grid_program = self.ctx.program(
            vertex_shader="""
                #version 330 core
                uniform mat4 vp;
                uniform vec3 gRoomCenter;
                uniform float gRoomSize;
                out vec3 v_pos;
                void main() {
                    const vec2 verts[4] = vec2[4](
                        vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, 1)
                    );
                    vec2 p = verts[gl_VertexID];
                    float h = gRoomSize * 0.5;
                    vec3 world = vec3(gRoomCenter.x + p.x * h, 0.0, gRoomCenter.z + p.y * h);
                    v_pos = world;
                    gl_Position = vp * vec4(world, 1.0);
                }
            """,
            fragment_shader="""
                #version 330 core
                in vec3 v_pos;
                out vec4 f_color;
                void main() {
                    vec2 p = v_pos.xz;
                    float h = 5.0;
                    vec2 g = abs(fract(p / h) - 0.5);
                    float line = min(g.x, g.y);
                    float alpha = 1.0 - smoothstep(0.0, 0.02, line) * 0.3;
                    f_color = vec4(0.3, 0.3, 0.3, alpha);
                }
            """,
        )
        self.grid_vao = self.ctx.vertex_array(self.grid_program, [])

    def _build_demo_instances(self):
        positions = [
            (2.0, 3.0, 1.0, np.array([1.0, 0.5, 0.15], dtype=np.float32)),
            (-2.0, 2.0, -1.0, np.array([0.2, 0.8, 0.9], dtype=np.float32)),
            (0.0, 4.0, 3.0, np.array([0.9, 0.2, 0.2], dtype=np.float32)),
            (-3.0, 1.5, 2.0, np.array([0.2, 0.9, 0.3], dtype=np.float32)),
            (3.0, 2.5, -2.0, np.array([0.8, 0.2, 0.8], dtype=np.float32)),
        ]
        self.num_instances = len(positions)
        self.instance_matrices = np.zeros((MAX_INSTANCES, 4, 4), dtype=np.float32)
        self.instance_colors = np.zeros((MAX_INSTANCES, 3), dtype=np.float32)
        for i, (px, py, pz, color) in enumerate(positions):
            r = INDICATOR_RADIUS / 4.0
            self.instance_matrices[i] = get_model_matrix(tx=px, ty=py, tz=pz, sx=r, sy=-py, sz=r)
            self.instance_colors[i] = color

    def load_program(self, vertex_shader, fragment_shader):
        with open(vertex_shader) as f:
            vs = f.read()
        with open(fragment_shader) as f:
            fs = f.read()
        return self.ctx.program(vertex_shader=vs, fragment_shader=fs)

    def _update_camera(self):
        self.eye[0] = self.target[0] + self._dist * np.cos(self._pitch) * np.sin(self._yaw)
        self.eye[1] = self.target[1] + self._dist * np.sin(self._pitch)
        self.eye[2] = self.target[2] + self._dist * np.cos(self._pitch) * np.cos(self._yaw)

    def get_view(self):
        return get_view_matrix(eye=self.eye, target=self.target)

    def get_vp(self):
        vp = get_perspective_matrix(self.fov, self.aspect_ratio, self.near, self.far)
        return vp @ self.get_view()

    def on_resize(self, width, height):
        self.aspect_ratio = width / height
        self.window_size = (width, height)

    def on_mouse_drag_event(self, x, y, dx, dy):
        self._yaw -= dx * 0.005
        self._pitch = np.clip(self._pitch + dy * 0.005, -np.pi / 2 + 0.01, np.pi / 2 - 0.01)
        self._update_camera()

    def on_mouse_scroll_event(self, x_offset, y_offset):
        self._dist = max(1.0, self._dist - y_offset * 0.5)
        self._update_camera()

    def on_render(self, time: float, frame_time: float):
        self.ctx.clear(0.1, 0.1, 0.15, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)

        vp = self.get_vp()
        self.cylinder_program["u_vp"].write(vp.astype("f4").T.tobytes())
        self.cylinder_program["u_models"].write(
            self.instance_matrices.transpose(0, 2, 1).ravel().astype("f4").tobytes()
        )
        self.cylinder_program["u_colors"].write(
            self.instance_colors.astype("f4").tobytes()
        )

        self.ctx.disable(moderngl.DEPTH_TEST)
        self.cylinder_vao.render(
            moderngl.TRIANGLES, CYLINDER_VERTS, instances=self.num_instances
        )

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.grid_program["vp"].write(vp.astype("f4").T.tobytes())
        self.grid_program["gRoomCenter"].write(np.array([0.0, 0.0, 0.0], dtype="f4").tobytes())
        self.grid_program["gRoomSize"].value = 10.0
        self.grid_vao.render(moderngl.TRIANGLE_STRIP, vertices=4)


if __name__ == "__main__":
    mglw.run_window_config(CylinderTestWindow)
