"""Legs through the resting object (DECISIONS.md D29), inside Blender, no rendering:
blender -b -P synth/scripts/leg_overlap.py -- OUT.jsonl [--apart] SCENARIO.json [SCENARIO.json ...]
Rebuilds the member's motion and object timeline (gen/build_scene.py, gen/motion.py). For every frame where the object
rests on the ground: horizontal penetration (m) of each walker's leg points (thigh, calf, foot, toe joints and 3 points
between them) into the object's oriented bounding box (z within its height). One JSON line per scenario: clip, outcome,
object (the asset, D41), events, per walker [t, depth] for frames with depth > 0, rest = the object's ground point
[x, y] at its first rest frame (null if it never rests), phases = the actor's max depth per motion phase
(qc.leg_phases), dis = the distractors' max depth, carry = [t, forward, left, z] per frame while the actor holds the
object (its ground point in the actor's frame and its bottom height; checks structures.carry_box,
run_batch.carry_excess, D38). Rule (D29, release v2): actor depth >= 0.08 m after t_walk_out (walks through the dropped object) or before
t_event (walks through a pre-existing one), or any distractor >= 0.08 m -> exclude. New batches (F1, D33): run_batch's
pre-render gate, qc.leg_verdict (0.05 m outside the crouch phases; in them qc.crouch_th of the object, D41).
--apart (F5, D32): also per rest frame [t, apart_px, object_px, actor_h]: the object seen apart from the actor without
rendering. Actor and object meshes (as evaluated: armature deformation) projected with the group's camera at the render
size and z-buffered per pixel; object_px = object pixels in front of the actor inside the camera image, apart_px = those
farther than qc.apart_radius(actor_h) from the actor's visible pixels (as composite.apart_px on the index pass;
distractors and parked vehicles are ignored), actor_h = the projected actor's pixel height."""
import sys, json, math, pathlib
import bpy, numpy as np
GEN = pathlib.Path(__file__).resolve().parents[1] / "gen"
sys.path.insert(0, str(GEN))
import build_scene as B, motion, tracks, qc
from camera import Camera, group_calib

LEG = ["Thigh", "Calf", "Foot", "Toe0"]


def leg_points(arm):
    pb = arm.pose.bones; mw = arm.matrix_world; P = []
    for s in "LR":
        js = [mw @ pb[f"Bip01 {s} {b}"].head for b in LEG if f"Bip01 {s} {b}" in pb]
        if f"Bip01 {s} Toe0" in pb:
            js.append(mw @ pb[f"Bip01 {s} Toe0"].tail)
        for a, b in zip(js[:-1], js[1:]):
            for w in (0.0, 0.25, 0.5, 0.75):
                P.append(a.lerp(b, w))
        P.append(js[-1])
    return np.array([[p.x, p.y, p.z] for p in P])


def mesh_tris(objs):
    """World-space triangles [N, 3, 3] of mesh objects as evaluated at the current frame."""
    dg = bpy.context.evaluated_depsgraph_get(); out = []
    for o in objs:
        eo = o.evaluated_get(dg); me = eo.to_mesh(); me.calc_loop_triangles()
        co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", co)
        tri = np.empty(len(me.loop_triangles) * 3, np.int64); me.loop_triangles.foreach_get("vertices", tri)
        M = np.array(eo.matrix_world)
        out.append((co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3])[tri.reshape(-1, 3)])
        eo.to_mesh_clear()
    return np.concatenate(out) if out else np.zeros((0, 3, 3))


def project(cam, T):
    """Triangles [N, 3, 3] -> [N, 3, 3] of (u, v, depth); triangles not wholly in front of the camera dropped."""
    uv, z = cam.project(T.reshape(-1, 3))
    P = np.c_[uv, z].reshape(-1, 3, 3)
    return P[(P[:, :, 2] > 0.1).all(1)]


def zbuffer(P, x0, y0, w, h, k=8):
    """Nearest depth per pixel of the grid with pixel centres (x0 + i + 0.5, y0 + j + 0.5); inf where no triangle.
    Triangles are rasterised on their pixel box, all at once on a k x k window where the box fits, else one by one."""
    zb = np.full(h * w, np.inf)
    lo, hi = P[:, :, :2].min(1), P[:, :, :2].max(1)
    i0 = np.clip(np.floor(lo[:, 0] - x0 - 0.5), 0, w).astype(int); i1 = np.clip(np.ceil(hi[:, 0] - x0 + 0.5), 0, w).astype(int)
    j0 = np.clip(np.floor(lo[:, 1] - y0 - 0.5), 0, h).astype(int); j1 = np.clip(np.ceil(hi[:, 1] - y0 + 0.5), 0, h).astype(int)
    a, b, c = P[:, 0], P[:, 1], P[:, 2]
    d = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    keep = (i1 > i0) & (j1 > j0) & (np.abs(d) > 1e-12)
    small = keep & (i1 - i0 <= k) & (j1 - j0 <= k)
    for sel, win in ((np.flatnonzero(small), k), *((np.array([t]), None) for t in np.flatnonzero(keep & ~small))):
        if not len(sel):
            continue
        I = i0[sel, None, None] + np.arange(win or i1[sel[0]] - i0[sel[0]])[None, None, :]
        J = j0[sel, None, None] + np.arange(win or j1[sel[0]] - j0[sel[0]])[None, :, None]
        I, J = np.broadcast_arrays(I, J)
        X, Y = x0 + I + 0.5, y0 + J + 0.5
        (ax, ay, az), (bx, by, bz), (cx, cy, cz) = [[v[sel, q][:, None, None] for q in range(3)] for v in (a, b, c)]
        D = d[sel][:, None, None]
        l1 = ((by - cy) * (X - cx) + (cx - bx) * (Y - cy)) / D
        l2 = ((cy - ay) * (X - cx) + (ax - cx) * (Y - cy)) / D
        l3 = 1 - l1 - l2
        ins = (l1 >= 0) & (l2 >= 0) & (l3 >= 0) & (I < i1[sel, None, None]) & (J < j1[sel, None, None])
        np.minimum.at(zb, (J * w + I)[ins], (l1 * az + l2 * bz + l3 * cz)[ins])
    return zb.reshape(h, w)


def dilate(m, r):
    """Binary dilation of m by qc.disk(r) (as cv2.dilate)."""
    out = np.zeros_like(m); H, W = m.shape
    for dy, dx in zip(*np.nonzero(qc.disk(r))):
        dy -= r; dx -= r
        out[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)] |= m[max(0, dy):H + min(0, dy), max(0, dx):W + min(0, dx)]
    return out


class Apart:
    """Per rest frame: object pixels seen apart from the actor (see the module docstring)."""

    def __init__(self, gr, arm, ob):
        self.cam = Camera(group_calib(gr), gr["render_size"]); self.W, self.H = gr["render_size"]
        self.meshes = [o for o in arm.children if o.type == "MESH"]; self.ob = ob
        self.R = qc.apart_radius(self.H) + 1  # grid margin: the largest dilation
        self.key = None

    def __call__(self):
        key = tuple(np.round(np.array(self.ob.matrix_world), 6).ravel())
        if key != self.key:  # the object moved: rasterise it again on a grid around it
            self.key = key
            P = project(self.cam, mesh_tris([self.ob]))
            if not len(P):
                self.grid = None
            else:
                lo, hi = P[:, :, :2].reshape(-1, 2).min(0), P[:, :, :2].reshape(-1, 2).max(0)
                x0, y0 = (int(max(0, np.floor(v - self.R))) for v in lo)
                x1, y1 = int(min(self.W, np.ceil(hi[0] + self.R))), int(min(self.H, np.ceil(hi[1] + self.R)))
                self.grid = (x0, y0, x1 - x0, y1 - y0) if x1 > x0 and y1 > y0 else None
                if self.grid:
                    self.oz = zbuffer(P, *self.grid)
                    X, Y = np.meshgrid(x0 + np.arange(x1 - x0) + 0.5, y0 + np.arange(y1 - y0) + 0.5)
                    self.inside = self.cam.in_image(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
        A = project(self.cam, mesh_tris(self.meshes))
        v = np.clip(A[:, :, 1], 0, self.H - 1) if len(A) else np.zeros(1)
        act_h = float(np.floor(v.max()) - np.floor(v.min()) + 1) if len(A) else 0.0
        if self.grid is None:
            return 0, 0, act_h
        az = zbuffer(A, *self.grid) if len(A) else np.full(self.oz.shape, np.inf)
        obj = (self.oz < az) & self.inside
        act = np.isfinite(az) & (az <= self.oz)
        apart = obj & ~dilate(act, qc.apart_radius(act_h)) if act.any() else obj
        return int(apart.sum()), int(obj.sum()), act_h


def carried(m, times, oloc):
    """[t, forward, left, z] per frame while the actor holds the object (state hand, or grasp after its 0.25 s blend):
    the object's ground point (bottom centre, build_scene.bake_object) in the actor's frame (root position and facing,
    motion.root_track), metres; checks structures.carry_box (C5 review G1)."""
    x, y, yaw = motion.root_track(m["root_keys"], times)
    st = sorted(m["object_states"], key=lambda s: s["t0"])
    out = []
    for k, t in enumerate(times):
        s = ([q for q in st if q["t0"] <= t] or st[:1])[-1]
        if s["state"] == "hand" or (s["state"] == "grasp" and t - s["t0"] >= 0.25):
            d = np.array(oloc[k][:2]) - (x[k], y[k]); c, sn = math.cos(yaw[k]), math.sin(yaw[k])
            out.append([round(float(t), 3), round(float(c * d[0] + sn * d[1]), 3), round(float(c * d[1] - sn * d[0]), 3),
                        round(float(oloc[k][2]), 3)])
    return out


def run(sc_path, apart=False):
    S = json.loads(pathlib.Path(sc_path).read_text()); gr, m = S["group"], S["member"]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene; fps = gr["render_fps"]
    sc.render.fps = max(1, int(round(fps))); sc.render.fps_base = sc.render.fps / fps
    n = int(math.floor(gr["duration"] * fps + 1e-6)); sc.frame_start, sc.frame_end = 1, n
    A = gr["actor"]
    arm, root = B.make_actor("actor", A["avatar"], A["gender"], A["scale"], 1)
    times = motion.bake(arm, root, m["segments"], m["root_keys"], A["gender"], fps, n, "actor")
    ob, dims = B.load_object(gr["object"]["asset"], gr["object"]["colour"], gr["object"].get("scale", 1.0))
    oloc, at_rest = B.bake_object(sc, ob, dims, arm, root, m["object_states"], times, gr["object"]["yaw"], 2, fps,
                                  gr["object"].get("setdown", 0.0))
    walkers = [("actor", arm)]
    for i, D in enumerate(gr["distractors"]):
        darm, droot = B.make_actor(f"dis{i}", D["avatar"], D["gender"], D["scale"], 3)
        segs = [dict(anim=D["walk"], t0=-120.0, t1=1e9, mode="loop")]
        keys = tracks.walker_keys(D["path"], D["t0"], D["speed"])
        motion.bake(darm, droot, segs, keys, D["gender"], fps, n, f"dis{i}")
        walkers.append((f"dis{i}", darm))
    V = np.array([v.co[:] for v in ob.data.vertices]); lo, hi = V.min(0), V.max(0)
    hits = {w: [] for w, _ in walkers}
    ap, est = (Apart(gr, arm, ob), []) if apart else (None, None)
    for k in range(n):
        if not at_rest[k]:
            continue
        sc.frame_set(k + 1)
        inv = np.array(ob.matrix_world.inverted())
        for w, a in walkers:
            P = leg_points(a); L = (inv[:3, :3] @ P.T).T + inv[:3, 3]
            inz = (L[:, 2] >= lo[2]) & (L[:, 2] <= hi[2])
            dx = np.minimum(L[:, 0] - lo[0], hi[0] - L[:, 0]); dy = np.minimum(L[:, 1] - lo[1], hi[1] - L[:, 1])
            d = np.where(inz, np.minimum(dx, dy), -1.0)
            if d.max() > 0:
                hits[w].append([round(float(times[k]), 3), round(float(d.max()), 3)])
        if ap is not None:
            est.append([round(float(times[k]), 3), *ap()])
    k0 = int(np.argmax(at_rest)) if at_rest.any() else None
    r = dict(clip=pathlib.Path(sc_path).stem, outcome=m["outcome"], object=gr["object"]["asset"],
             dims=[round(float(x), 3) for x in dims],
             rest_frames=int(at_rest.sum()), events=m["events"], hits=hits,
             rest=None if k0 is None else [round(float(oloc[k0][0]), 4), round(float(oloc[k0][1]), 4)],
             phases=qc.leg_phases(m["events"], m["outcome"], hits["actor"]),
             dis=max([d for w, h in hits.items() if w != "actor" for _, d in h] or [0.0]),
             carry=carried(m, times, oloc))
    if apart:
        r.update(fps=fps, apart=est)
    return r


args = sys.argv[sys.argv.index("--") + 1:]
out = pathlib.Path(args[0]); APART = "--apart" in args
with out.open("a") as f:
    for p in [a for a in args[1:] if a != "--apart"]:
        try:
            r = run(p, APART)
        except Exception as e:
            r = dict(clip=pathlib.Path(p).stem, error=repr(e))
        f.write(json.dumps(r) + "\n"); f.flush()
