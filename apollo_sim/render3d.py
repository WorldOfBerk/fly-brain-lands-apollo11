"""3D landing view with moderngl: terrain, LM model, LM shadow, dust.

  - Terrain: the apollo_sim.terrain height map (same as the physics), baked sun shadows, procedural
    fine regolith in the shader. Lunar curvature in the distance: y -= d^2 / 2R.
  - Photometry: Lommel-Seeliger + Lambert mix with an opposition surge at zero phase (the bright halo
    around the LM shadow in the Apollo 11 film).
  - Sun at 10.8 deg in the east. Black sky (no stars in the Apollo cameras).
  - The DPS plume is nearly invisible in vacuum, so none is drawn. Radial dust sheet below ~15 m.
  - LM: main shapes of Eagle (octagonal gold-foil descent stage, ascent stage, four legs, pads,
    1.7 m probes on three legs, nozzle, RCS quads), approximate dimensions.
"""
import math

import moderngl
import numpy as np

from .terrain import SUN_DIR, Terrain

R_MOON = 1_737_400.0


# ---------------------------------------------------------------- math
def perspective(fovy, aspect, near, far):
    f = 1 / math.tan(fovy / 2)
    return np.array([[f / aspect, 0, 0, 0], [0, f, 0, 0],
                     [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)], [0, 0, -1, 0]], np.float32)


def ortho(l, r, b, t, n, f):
    return np.array([[2 / (r - l), 0, 0, -(r + l) / (r - l)], [0, 2 / (t - b), 0, -(t + b) / (t - b)],
                     [0, 0, -2 / (f - n), -(f + n) / (f - n)], [0, 0, 0, 1]], np.float32)


def look_at(eye, target, up=(0, 1, 0)):
    eye, target, up = map(lambda v: np.asarray(v, np.float64), (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m.astype(np.float32)


def rot_z(a):  # pitch, in the downrange (x) - up (y) plane
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)


def translate(x, y, z):
    m = np.eye(4, dtype=np.float32)
    m[:3, 3] = (x, y, z)
    return m


# ---------------------------------------------------------------- LM geometry
def _cyl(r0, r1, y0, y1, n=16, sides=True, caps=True, cx=0.0, cz=0.0, phase=0.0):
    tris = []
    a = np.linspace(0, 2 * np.pi, n + 1) + phase
    for i in range(n):
        p = [(cx + r0 * math.cos(a[i]), y0, cz + r0 * math.sin(a[i])), (cx + r0 * math.cos(a[i + 1]), y0, cz + r0 * math.sin(a[i + 1])),
             (cx + r1 * math.cos(a[i + 1]), y1, cz + r1 * math.sin(a[i + 1])), (cx + r1 * math.cos(a[i]), y1, cz + r1 * math.sin(a[i]))]
        if sides:
            tris += [p[0], p[2], p[1], p[0], p[3], p[2]]
        if caps:
            tris += [(cx, y0, cz), p[1], p[0], (cx, y1, cz), p[3], p[2]]
    return tris


def _box(x0, x1, y0, y1, z0, z1):
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    f = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (3, 7, 6), (3, 6, 2), (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5)]
    return [v[i] for t in f for i in t]


def _rod(p, q, r=0.06, n=8):
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    L = np.linalg.norm(d)
    z = d / L
    x = np.cross(z, [0, 1, 0] if abs(z[1]) < 0.9 else [1, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    out = []
    for i in range(n):
        a0, a1 = 2 * np.pi * i / n, 2 * np.pi * (i + 1) / n
        o0, o1 = r * (np.cos(a0) * x + np.sin(a0) * y), r * (np.cos(a1) * x + np.sin(a1) * y)
        out += [tuple(p + o0), tuple(q + o1), tuple(p + o1), tuple(p + o0), tuple(q + o0), tuple(q + o1)]
    return out


GOLD, BLACK, SILVER, GREY, DARK = (0.72, 0.46, 0.10), (0.07, 0.07, 0.08), (0.78, 0.79, 0.80), (0.45, 0.46, 0.48), (0.2, 0.2, 0.22)


def lm_mesh():
    """Triangles as (pos, normal, colour, specular). Origin at the pad bottoms; y up, x downrange."""
    parts = []  # (triangles, colour, specular)
    # descent stage: octagonal prism, 4.2 m wide, ~1.6-3.25 m above the pads
    parts.append((_cyl(2.15, 2.15, 1.55, 3.20, n=8, phase=math.pi / 8), GOLD, 0.9))
    parts.append((_cyl(0.95, 0.55, 0.65, 1.55, n=20, caps=False), DARK, 0.3))  # DPS nozzle
    # ascent stage
    parts.append((_box(-1.15, 1.25, 3.20, 5.60, -1.35, 1.35), SILVER, 0.6))  # cabin
    parts.append((_box(1.25, 1.85, 3.40, 5.10, -1.05, 1.05), GREY, 0.4))  # front
    parts.append((_box(-1.55, -1.15, 3.30, 5.00, -1.10, 1.10), BLACK, 0.2))  # aft equipment bay
    parts.append((_cyl(0.70, 0.70, 3.30, 4.60, n=14, cx=-0.2, cz=1.75), SILVER, 0.6))  # side tank
    parts.append((_cyl(0.70, 0.70, 3.30, 4.60, n=14, cx=-0.2, cz=-1.75), SILVER, 0.6))
    parts.append((_cyl(0.42, 0.42, 5.60, 6.20, n=14), GREY, 0.5))  # tunnel
    parts.append((_cyl(0.05, 0.55, 6.70, 6.95, n=14, cx=0.6), SILVER, 0.7))  # rendezvous radar
    parts.append((_box(1.86, 1.92, 4.10, 4.75, -0.95, -0.35), BLACK, 0.9))  # window
    parts.append((_box(1.86, 1.92, 4.10, 4.75, 0.35, 0.95), BLACK, 0.9))
    for sx, sz in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:  # RCS quads
        parts.append((_box(0.9 * sx - 0.18, 0.9 * sx + 0.18, 5.0, 5.4, 1.7 * sz - 0.18, 1.7 * sz + 0.18), BLACK, 0.3))
    # legs on the diagonals, pads at 4.6 m radius
    legs, pads, probes = [], [], []
    for k, ang in enumerate([45, 135, 225, 315]):
        a = math.radians(ang)
        ca, sa = math.cos(a), math.sin(a)
        top = (2.05 * ca, 3.0, 2.05 * sa)
        mid = (2.05 * ca, 1.7, 2.05 * sa)
        foot = (4.55 * ca, 0.32, 4.55 * sa)
        legs += _rod(top, foot, 0.09) + _rod(mid, foot, 0.05)
        pads += _cyl(0.47, 0.40, 0.0, 0.32, n=16, cx=foot[0], cz=foot[2])
        if k != 2:  # no probe on the ladder leg (Apollo 11)
            probes += _rod((foot[0], 0.0, foot[2]), (foot[0], -1.7, foot[2]), 0.025, 6)
    parts += [(legs, GOLD, 0.8), (pads, SILVER, 0.5), (probes, GREY, 0.4)]
    data = []
    for tris, col, spec in parts:
        t = np.asarray(tris, np.float32).reshape(-1, 3, 3)
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
        for tri, nn in zip(t, n):
            for v in tri:
                data.append((*v, *nn, *col, spec))
    return np.asarray(data, np.float32)


# ---------------------------------------------------------------- shaders
TERRAIN_VS = """
#version 330
in vec2 in_uv;            // [-1,1]^2 grid
uniform vec2 center;       // camera centre (x, z)
uniform float extent;      // half width (m)
uniform vec4 bounds;       // x_min, x_max, z_min, z_max
uniform sampler2D hmap;
uniform vec3 cam;
uniform mat4 mvp;
uniform mat4 sun_mvp;
out vec3 v_pos; out vec2 v_tex; out vec4 v_sun;
void main() {
    vec2 q = sign(in_uv) * (0.06 * abs(in_uv) + 0.94 * in_uv * in_uv);   // denser near the centre
    vec2 xz = center + q * extent;
    vec2 tex = vec2((xz.x - bounds.x) / (bounds.y - bounds.x), (xz.y - bounds.z) / (bounds.w - bounds.z));
    // off-map: repeat in x, mirror in z (keeps shadows consistent with a sun on the x axis)
    tex = vec2(fract(tex.x), 1.0 - abs(1.0 - mod(tex.y, 2.0)));
    float h = texture(hmap, tex).r;
    float d2 = dot(xz - cam.xz, xz - cam.xz);
    vec3 p = vec3(xz.x, h - d2 / (2.0 * 1737400.0), xz.y);
    v_pos = p; v_tex = tex; v_sun = sun_mvp * vec4(p, 1.0);
    gl_Position = mvp * vec4(p, 1.0);
}
"""

TERRAIN_FS = """
#version 330
in vec3 v_pos; in vec2 v_tex; in vec4 v_sun;
uniform sampler2D nmap; uniform sampler2D smap; uniform sampler2D amap; uniform sampler2D lm_shadow;
uniform vec3 sun; uniform vec3 cam; uniform float exposure; uniform float pixel_angle;
out vec4 color;
float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p){ vec2 i = floor(p), f = fract(p); vec2 u = f*f*(3.0-2.0*f);
  return mix(mix(hash(i), hash(i+vec2(1,0)), u.x), mix(hash(i+vec2(0,1)), hash(i+vec2(1,1)), u.x), u.y); }
void main() {
    vec3 n = normalize(texture(nmap, v_tex).xyz * 2.0 - 1.0);
    float lit = 1.0 - smoothstep(0.0, 0.3, texture(smap, v_tex).r);
    float alb = texture(amap, v_tex).r;
    // fine regolith, band-limited to the pixel footprint (no aliasing)
    float dist = length(cam - v_pos);
    float pix = dist * pixel_angle / max(abs(normalize(cam - v_pos).y), 0.15);
    float fine = 0.0; vec2 g = vec2(0.0);
    for (int o = 0; o < 5; o++) {
        float lam = 0.15 * pow(3.0, float(o));           // m
        float w = clamp(lam / (4.0 * pix) - 0.5, 0.0, 1.0);
        fine += w * (noise(v_pos.xz / lam) - 0.5) * 0.45;
        g += w * vec2(noise(v_pos.xz / lam + 17.0) - 0.5, noise(v_pos.xz / lam + 43.0) - 0.5) * 0.3;
    }
    n = normalize(n + vec3(g.x, 0.0, g.y));
    vec3 v = normalize(cam - v_pos);
    float mu0 = max(dot(n, sun), 0.0), mu = max(dot(n, v), 1e-3);
    float ls = mu0 / (mu0 + mu);                       // Lommel-Seeliger
    float brdf = mix(2.0 * ls, mu0, 0.35);
    float phase = acos(clamp(dot(v, sun), -1.0, 1.0));
    float surge = 1.0 + 0.9 * exp(-phase / 0.06);      // opposition surge
    // LM shadow map
    vec3 sp = v_sun.xyz / v_sun.w * 0.5 + 0.5;
    float lmsh = 0.0;
    if (all(greaterThan(sp.xy, vec2(0.0))) && all(lessThan(sp.xy, vec2(1.0)))) {
        float dmin = texture(lm_shadow, sp.xy).r;
        lmsh = sp.z - 0.002 > dmin ? 1.0 : 0.0;
    }
    float L = alb * (1.0 + fine) * brdf * surge * lit * (1.0 - lmsh) / 0.085 * 0.33;
    L = 1.0 - exp(-L * exposure);
    color = vec4(vec3(pow(L, 1.0 / 2.2)) * vec3(1.0, 0.985, 0.96), 1.0);
}
"""

LM_VS = """
#version 330
in vec3 in_pos; in vec3 in_norm; in vec3 in_col; in float in_spec;
uniform mat4 mvp; uniform mat4 model;
out vec3 v_n; out vec3 v_col; out float v_spec; out vec3 v_pos;
void main() { v_n = mat3(model) * in_norm; v_col = in_col; v_spec = in_spec;
  v_pos = (model * vec4(in_pos, 1.0)).xyz; gl_Position = mvp * vec4(in_pos, 1.0); }
"""

LM_FS = """
#version 330
in vec3 v_n; in vec3 v_col; in float v_spec; in vec3 v_pos;
uniform vec3 sun; uniform vec3 cam; uniform float exposure;
out vec4 color;
float hash(vec3 p){ return fract(sin(dot(p, vec3(12.9898, 78.233, 37.719))) * 43758.5453); }
void main() {
    vec3 n = normalize(v_n);
    if (v_spec > 0.85) n = normalize(n + 0.25 * (vec3(hash(floor(v_pos * 9.0)), hash(floor(v_pos * 9.0) + 3.1), hash(floor(v_pos * 9.0) + 7.7)) - 0.5)); // crinkled foil
    vec3 v = normalize(cam - v_pos);
    float d = max(dot(n, sun), 0.0);
    float s = pow(max(dot(reflect(-sun, n), v), 0.0), 40.0) * v_spec * 3.0;
    vec3 ground_bounce = vec3(0.06) * max(-n.y, 0.0) + 0.03;  // light bounced off the surface
    vec3 c = v_col * (d * 1.6 + ground_bounce) + s * mix(vec3(1.0), v_col, 0.6);
    c = 1.0 - exp(-c * exposure);
    color = vec4(pow(c, vec3(1.0 / 2.2)), 1.0);
}
"""

DEPTH_VS = """
#version 330
in vec3 in_pos; in vec3 in_norm; in vec3 in_col; in float in_spec;
uniform mat4 mvp;
void main() { gl_Position = mvp * vec4(in_pos, 1.0); }
"""
DEPTH_FS = "#version 330\nvoid main() {}\n"

DUST_VS = """
#version 330
in vec3 in_pos; in float in_a;
uniform mat4 mvp; uniform float psize;
out float v_a;
void main() { v_a = in_a; gl_Position = mvp * vec4(in_pos, 1.0); gl_PointSize = psize / gl_Position.w; }
"""
DUST_FS = """
#version 330
in float v_a; out vec4 color;
void main() { vec2 c = gl_PointCoord - 0.5; float r = dot(c, c);
  if (r > 0.25) discard; color = vec4(vec3(0.85, 0.84, 0.82), v_a * (1.0 - 4.0 * r)); }
"""


class Renderer:
    def __init__(self, terrain: Terrain, width=1280, height=720, ctx=None):
        self.T = terrain
        self.W, self.H = width, height
        self.ctx = ctx or moderngl.create_standalone_context()
        ctx = self.ctx
        s = terrain.s
        self.bounds = (s.x_min, s.x_max, s.z_min, s.z_max)
        # textures: rows z, columns x -> (u = x, v = z)
        self.t_h = ctx.texture((terrain.nx, terrain.nz), 1, terrain.h.astype("f4").tobytes(), dtype="f4")
        nrm = ((terrain.normal * 0.5 + 0.5) * 255).astype("u1")
        self.t_n = ctx.texture((terrain.nx, terrain.nz), 3, nrm.tobytes())
        # shadow depth (m); linear filtering gives sub-pixel soft edges
        self.t_s = ctx.texture((terrain.nx, terrain.nz), 1, terrain.shadow_depth.astype("f4").tobytes(), dtype="f4")
        self.t_a = ctx.texture((terrain.nx, terrain.nz), 1, terrain.albedo.astype("f4").tobytes(), dtype="f4")
        for t in (self.t_h, self.t_n, self.t_s, self.t_a):
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = t.repeat_y = True
        for t in (self.t_n, self.t_s, self.t_a):  # mipmaps + anisotropic filtering in the distance
            t.build_mipmaps()
            t.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            t.anisotropy = 16.0
        self.prog_t = ctx.program(vertex_shader=TERRAIN_VS, fragment_shader=TERRAIN_FS)
        n = 700
        u = np.linspace(-1, 1, n, dtype="f4")
        U, V = np.meshgrid(u, u)
        grid = np.stack([U, V], -1).reshape(-1, 2)
        idx = np.arange(n * n).reshape(n, n)
        tri = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1)
        self.vao_t = ctx.vertex_array(self.prog_t, [(ctx.buffer(grid.tobytes()), "2f", "in_uv")],
                                      index_buffer=ctx.buffer(tri.astype("i4").tobytes()))
        # LM
        self.prog_lm = ctx.program(vertex_shader=LM_VS, fragment_shader=LM_FS)
        self.prog_depth = ctx.program(vertex_shader=DEPTH_VS, fragment_shader=DEPTH_FS)
        vb = ctx.buffer(lm_mesh().tobytes())
        self.vao_lm = ctx.vertex_array(self.prog_lm, [(vb, "3f 3f 3f 1f", "in_pos", "in_norm", "in_col", "in_spec")])
        self.vao_lm_d = ctx.vertex_array(self.prog_depth, [(vb, "3f 28x", "in_pos")])
        # LM shadow map
        self.shadow_tex = ctx.depth_texture((2048, 2048))
        self.shadow_fbo = ctx.framebuffer(depth_attachment=self.shadow_tex)
        # dust
        self.prog_d = ctx.program(vertex_shader=DUST_VS, fragment_shader=DUST_FS)
        self.dust_n = 20000
        self.dust_buf = ctx.buffer(reserve=self.dust_n * 16)
        self.vao_dust = ctx.vertex_array(self.prog_d, [(self.dust_buf, "3f 1f", "in_pos", "in_a")])
        self.rng = np.random.default_rng(3)
        self.dust = np.zeros((0, 7), np.float32)  # x y z vx vy vz age
        self.fbo = ctx.framebuffer(color_attachments=[ctx.texture((width, height), 4)],
                                   depth_attachment=ctx.depth_texture((width, height)))

    def _update_dust(self, lm_pos, alt, thrust_frac, dt):
        if alt < 15.0 and thrust_frac > 0.02:
            k = int(1200 * thrust_frac * (1 - alt / 15.0) * dt / 0.05)
            a = self.rng.uniform(0, 2 * np.pi, k)
            r0 = self.rng.uniform(0.5, 4.0, k)
            sp = self.rng.uniform(15, 60, k) * (1 - alt / 18.0)
            x, z = lm_pos[0] + r0 * np.cos(a), lm_pos[2] + r0 * np.sin(a)
            y = self.T.height(x, z) + self.rng.uniform(0.0, 0.25, k)
            new = np.stack([x, y, z, sp * np.cos(a), self.rng.uniform(0, 0.6, k), sp * np.sin(a), np.zeros(k)], 1)
            self.dust = np.concatenate([self.dust, new.astype(np.float32)])[-self.dust_n:]
        if len(self.dust):
            self.dust[:, 0:3] += self.dust[:, 3:6] * dt  # ballistic in vacuum
            self.dust[:, 4] -= 1.62 * dt
            self.dust[:, 6] += dt
            self.dust = self.dust[self.dust[:, 6] < 1.2]

    def render(self, lm_x, lm_y, pitch, thrust_frac=0.0, camera="chase", dt=0.05, exposure=1.3):
        """lm_x: downrange, lm_y: absolute height (same datum as the height map)."""
        ctx = self.ctx
        lm_pos = np.array([lm_x, lm_y, 0.0])
        model = translate(*lm_pos) @ rot_z(-pitch)  # pitch > 0 tilts the body forward
        alt = lm_y - float(self.T.height(lm_x, 0.0))
        if camera == "window":  # out of Aldrin's right window, like the 16 mm DAC camera
            eye = (model @ np.array([2.05, 4.45, 0.65, 1.0]))[:3]
            fwd = (model @ np.array([1.0, -0.9, 0.35, 0.0]))[:3]
            view = look_at(eye, eye + fwd)
            fov = math.radians(55)
            proj = perspective(fov, self.W / self.H, 0.3, 30000.0)
        else:
            back = 26 + 0.15 * min(alt, 400)
            eye = lm_pos + np.array([-back * 0.85, 4 + 0.35 * min(alt, 200) + 3, -back * 0.55])
            gy = float(self.T.height(eye[0], eye[2]))
            eye[1] = max(eye[1], gy + 2.0)
            # look further down when high so the ground stays in view
            view = look_at(eye, lm_pos + np.array([6.0, 2.0 - 0.5 * min(alt, 20000) ** 0.75, 0.0]))
            fov = math.radians(50)
            proj = perspective(fov, self.W / self.H, 0.5, 30000.0)
        vp = proj @ view
        # sun shadow map for the LM
        sun = SUN_DIR.astype(np.float32)
        sun_view = look_at(lm_pos + sun * 200.0, lm_pos, up=(0, 0, 1))
        sun_proj = ortho(-9, 9, -9, 9, 1.0, 600.0)
        sun_vp = sun_proj @ sun_view
        self.shadow_fbo.use()
        self.shadow_fbo.clear(depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        self.prog_depth["mvp"].write((sun_vp @ model).T.copy().tobytes())
        self.vao_lm_d.render()
        # scene
        self.fbo.use()
        self.fbo.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        p = self.prog_t
        p["center"].value = (float(eye[0]), float(eye[2]))
        p["extent"].value = float(max(6000.0, 6.0 * alt))
        p["bounds"].value = self.bounds
        p["cam"].value = tuple(map(float, eye))
        p["mvp"].write(vp.T.copy().tobytes())
        p["sun_mvp"].write(sun_vp.T.copy().tobytes())
        p["sun"].value = tuple(map(float, sun))
        p["exposure"].value = exposure
        p["pixel_angle"].value = float(fov / self.H)
        for i, (name, tex) in enumerate([("hmap", self.t_h), ("nmap", self.t_n), ("smap", self.t_s), ("amap", self.t_a),
                                         ("lm_shadow", self.shadow_tex)]):
            tex.use(i)
            p[name].value = i
        self.vao_t.render()
        q = self.prog_lm
        q["mvp"].write((vp @ model).T.copy().tobytes())
        q["model"].write(model.T.copy().tobytes())
        q["sun"].value = tuple(map(float, sun))
        q["cam"].value = tuple(map(float, eye))
        q["exposure"].value = exposure
        self.vao_lm.render()
        # dust
        self._update_dust(lm_pos, alt, thrust_frac, dt)
        if len(self.dust):
            ctx.enable(moderngl.BLEND)
            ctx.enable(moderngl.PROGRAM_POINT_SIZE)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            age = self.dust[:, 6]
            a = np.clip(0.12 * (1 - age / 1.2), 0, 1).astype(np.float32)
            buf = np.concatenate([self.dust[:, :3], a[:, None]], 1).astype("f4")
            self.dust_buf.orphan(self.dust_n * 16)
            self.dust_buf.write(buf.tobytes())
            self.prog_d["mvp"].write(vp.T.copy().tobytes())
            self.prog_d["psize"].value = 120.0
            self.vao_dust.render(moderngl.POINTS, vertices=len(buf))
            ctx.disable(moderngl.BLEND)
        img = np.frombuffer(self.fbo.read(components=3), np.uint8).reshape(self.H, self.W, 3)
        return img[::-1]
