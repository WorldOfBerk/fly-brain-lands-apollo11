"""Brain activity view: MaleCNS neurons drawn at their real cell-body positions (dorsal view, brain on
top, VNC below); active neurons light up. Additive point cloud in moderngl.

Positions: body-annotations somaLocation (8 nm voxels -> um). Neurons whose soma lies outside the
imaged volume (mostly sensory) have no position and are not drawn, but they are simulated.
"""
import math

import moderngl
import numpy as np

VS = """
#version 330
in vec3 in_pos; in float in_act; in vec3 in_col;
uniform mat4 mvp; uniform float base_size;
out float v_act; out vec3 v_col;
void main() {
    v_act = in_act; v_col = in_col;
    gl_Position = mvp * vec4(in_pos, 1.0);
    gl_PointSize = base_size * (1.0 + 3.0 * in_act);
}
"""
FS = """
#version 330
in float v_act; in vec3 v_col; out vec4 color;
void main() {
    vec2 c = gl_PointCoord - 0.5; float r2 = dot(c, c);
    if (r2 > 0.25) discard;
    float fall = exp(-r2 * 14.0);
    vec3 base = vec3(0.16, 0.20, 0.28) * 0.30;
    vec3 hot = mix(v_col, vec3(1.0, 0.97, 0.9), clamp(v_act - 0.6, 0.0, 1.0));
    color = vec4((base + hot * v_act * 1.6) * fall, 1.0);
}
"""


class BrainView:
    def __init__(self, soma_um, colors, width=640, height=1000, ctx=None):
        """soma_um: (n, 3) in um, NaN = unknown; colors: (n, 3) glow colour per neuron."""
        self.ctx = ctx or moderngl.create_standalone_context()
        self.W, self.H = width, height
        ok = np.isfinite(soma_um[:, 0])
        self.idx = np.flatnonzero(ok)
        P = soma_um[ok].astype(np.float64)
        self.center = 0.5 * (P.min(0) + P.max(0))
        P = P - self.center
        self.pos = P.astype("f4")
        self.prog = self.ctx.program(vertex_shader=VS, fragment_shader=FS)
        self.vb_pos = self.ctx.buffer(self.pos.tobytes())
        self.vb_act = self.ctx.buffer(reserve=len(self.idx) * 4)
        self.vb_col = self.ctx.buffer(colors[ok].astype("f4").tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [(self.vb_pos, "3f", "in_pos"), (self.vb_act, "1f", "in_act"),
                                                     (self.vb_col, "3f", "in_col")])
        self.fbo = self.ctx.framebuffer(color_attachments=[self.ctx.texture((width, height), 4)])
        self.act = np.zeros(len(self.idx), np.float32)

    def matrix(self, yaw):
        """Dorsal view: axis 0 (left-right) horizontal, axis 2 (anterior-posterior) vertical, brain on top."""
        c, s = math.cos(yaw), math.sin(yaw)
        # slight rotation about the long axis for parallax, then project
        R = np.array([[c, s, 0, 0], [-s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        span = 560.0  # um, half height
        aspect = self.W / self.H
        proj = np.array([[1 / (span * aspect), 0, 0, 0],
                         [0, 0, -1 / span, 0],
                         [0, 1 / 2000.0, 0, 0],
                         [0, 0, 0, 1]], np.float32)
        return proj @ R

    def project(self, P_um, yaw):
        """Pixel coordinates (for labels)."""
        m = self.matrix(yaw)
        p = (np.c_[P_um - self.center, np.ones(len(P_um))] @ m.T)[:, :2]
        return np.c_[(p[:, 0] * 0.5 + 0.5) * self.W, (1 - (p[:, 1] * 0.5 + 0.5)) * self.H]

    def update(self, rates_hz, dt, tau=0.25):
        """rates_hz for all neurons. Brightness ~ 1 - exp(-r/40), smoothed in time."""
        target = 1.0 - np.exp(-rates_hz[self.idx] / 40.0)
        a = 1 - math.exp(-dt / tau)
        self.act += a * (target.astype(np.float32) - self.act)

    def render(self, yaw=0.0):
        self.fbo.use()
        self.fbo.clear(0.0, 0.0, 0.0, 1.0)
        self.ctx.enable(moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE
        self.vb_act.write(self.act.tobytes())
        self.prog["mvp"].write(self.matrix(yaw).T.copy().tobytes())
        self.prog["base_size"].value = 2.0
        self.vao.render(moderngl.POINTS)
        self.ctx.disable(moderngl.BLEND)
        img = np.frombuffer(self.fbo.read(components=3), np.uint8).reshape(self.H, self.W, 3)
        return img[::-1]
