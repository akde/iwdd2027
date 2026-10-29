"""Render one scenario member (inside Blender):
blender -b -P synth/gen/build_scene.py -- SCENARIO.json OUT_DIR [--samples 24] [--frames a:b]
Writes OUT_DIR/rgba_####.png (8-bit RGBA, straight alpha), shadow_####.png (16-bit, Shadow Catcher pass, 1 = unlit
by nothing), index_####.png (8-bit, object pass index * 25: 1 actor, 2 object, 3 distractor, 4 distractor object)
for frames whose foreground can be in view, and OUT_DIR/meta.json (per frame: rendered flag, border, actor head
pixel + radius, actor foot/top pixels, object rest flag). Frame k+1 is clip time k / fps.
Only the actor, the object, distractors and their shadows are rendered; the plate is composited later."""
import sys, json, math, pathlib
import bpy, mathutils
import numpy as np

GEN = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(GEN))
import rb, motion
from camera import Camera, group_calib
from tracks import walker_keys

SYN = GEN.parent
G = 9.81
SKY_R = 2.0  # m: the sky light's shadow on the ground reaches about 2 m from a person (D39): the render region holds it
# set-down distance (D41): an object with gr['object']['setdown'] = d > 0 is put down d m further forward (the actor's
# facing) than under the palm; the offset grows with the hand's descent, 0 above SET_HI x the actor's scale (walking,
# standing: palm >= 0.735 x scale) and d below SET_LO x scale (the bottom of every bend: palm 0.378 x scale), so the
# released, the pre-existing and the re-grasped object all lie at scenario.rest_box; d = 0: the placement before D41
SET_HI, SET_LO = 0.70, 0.45


def args():
    a = sys.argv[sys.argv.index("--") + 1:]
    opt = dict(samples=24, frames=None)
    if "--samples" in a:
        opt["samples"] = int(a[a.index("--samples") + 1])
    if "--frames" in a:
        opt["frames"] = [int(x) for x in a[a.index("--frames") + 1].split(":")]
    return pathlib.Path(a[0]), pathlib.Path(a[1]), opt


def setup(gr, cal, fps, samples, out):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"; prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == "OPTIX"
    sc.cycles.device = "GPU"; sc.cycles.samples = samples; sc.cycles.use_denoising = True
    sc.cycles.denoiser = "OPTIX"; sc.cycles.seed = gr["seed"] % 100000
    sc.render.use_persistent_data = True
    W, H = gr["render_size"]
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = W, H, 100
    sc.render.film_transparent = True
    sc.view_settings.view_transform = "Standard"
    sc.render.use_motion_blur = True; sc.render.motion_blur_shutter = 0.5
    sc.render.fps = max(1, int(round(fps)))  # Blender ignores fps > 120: integer rate + base (14.985 -> 15 / 1.001)
    sc.render.fps_base = sc.render.fps / fps
    sc.render.use_border = True; sc.render.use_crop_to_border = False
    # camera
    M = mathutils.Matrix(cal["camera_world"])
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam)
    cam.matrix_world = M
    cam.data.sensor_fit = "HORIZONTAL"; cam.data.sensor_width = 36.0
    cam.data.lens = cal["focal_px"] * W / cal["image_size"][0] * 36.0 / W
    cam.data.clip_start, cam.data.clip_end = 0.1, 1000
    sc.camera = cam
    # ground: shadow catcher
    bpy.ops.mesh.primitive_plane_add(size=600, location=(cal["camera_world"][0][3], cal["camera_world"][1][3], 0))
    gnd = bpy.context.object; gnd.is_shadow_catcher = True; gnd.name = "ground"
    # light
    L = gr["light"]
    world = bpy.data.worlds.new("w"); sc.world = world; world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    sky = (0.55, 0.6, 0.7) if not L["night"] else (0.3, 0.3, 0.35)
    # night without a lamp: the sky is the only light and sets just the level (the compositor's gain matches the plate);
    # at 0.08 the avatar sat at 1-6 of 255 in the 8-bit pass and needed gains near 19 (D39)
    bg.inputs[0].default_value = (*sky, 1)
    bg.inputs[1].default_value = 0.08 if L["night"] and L.get("lamp") else (0.7 if L["sun"] else 1.6)
    if L["sun"]:
        sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN")); sc.collection.objects.link(sun)
        sun.data.energy = 3.5; sun.data.angle = math.radians(1.0)
        az, el = math.radians(L["sun_az"]), math.radians(L["sun_el"])
        d = mathutils.Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))  # towards sun
        sun.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
    lamp_loc = None
    if L["night"] and L.get("lamp"):
        X = gr["path"]["X"]
        lamp = bpy.data.objects.new("lamp", bpy.data.lights.new("lamp", "POINT")); sc.collection.objects.link(lamp)
        rng = np.random.default_rng(gr["seed"])
        lamp.location = (X[0] + rng.uniform(-5, 5), X[1] + rng.uniform(-5, 5), rng.uniform(4, 6))
        lamp.data.energy = 1500; lamp.data.shadow_soft_size = 0.3
        lamp_loc = np.array(lamp.location)
    # passes -> files
    vl = sc.view_layers[0]
    vl.cycles.use_pass_shadow_catcher = True; vl.use_pass_object_index = True
    sc.use_nodes = True
    nt = sc.node_tree; rl = nt.nodes["Render Layers"]
    for n in [n for n in nt.nodes if n.type == "COMPOSITE"]:
        nt.nodes.remove(n)

    def fout(name, sock, depth, mode):
        fo = nt.nodes.new("CompositorNodeOutputFile"); fo.base_path = str(out)
        fo.format.file_format = "PNG"; fo.format.color_depth = depth; fo.format.color_mode = mode
        fo.format.compression = 15
        fo.file_slots[0].path = name + "_"
        nt.links.new(sock, fo.inputs[0])
    fout("rgba", rl.outputs["Image"], "8", "RGBA")
    fout("shadow", rl.outputs["Shadow Catcher"], "16", "BW")
    div = nt.nodes.new("CompositorNodeMath"); div.operation = "DIVIDE"; div.inputs[1].default_value = 10.0
    nt.links.new(rl.outputs["IndexOB"], div.inputs[0])
    fout("index", div.outputs[0], "8", "BW")
    comp = nt.nodes.new("CompositorNodeComposite"); nt.links.new(rl.outputs["Image"], comp.inputs[0])
    return sc, lamp_loc


def load_object(asset, colour=None, scale=1.0):
    lib = SYN / "assets/objects/library.blend"
    with bpy.data.libraries.load(str(lib), link=False) as (src, dst):
        dst.objects = [asset]
    ob = dst.objects[0]
    bpy.context.scene.collection.objects.link(ob)
    ob.use_fake_user = False
    if colour is not None and asset in ("trashbag", "compost_bag_02"):
        for i, m in enumerate(ob.data.materials):
            m = m.copy(); ob.data.materials[i] = m
            bsdf = next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
            if bsdf is not None:
                for l in list(bsdf.inputs["Base Color"].links):
                    m.node_tree.links.remove(l)
                bsdf.inputs["Base Color"].default_value = (*colour, 1)
    if scale != 1.0:
        for v in ob.data.vertices:
            v.co *= scale
    dims = [max(v.co[i] for v in ob.data.vertices) - min(v.co[i] for v in ob.data.vertices) for i in range(3)]
    return ob, dims


def make_actor(name, avatar, gender, scale, index):
    arm = rb.import_avatar(avatar)
    arm.name = name
    root = bpy.data.objects.new(name + "_root", None); bpy.context.scene.collection.objects.link(root)
    arm.parent = root; root.scale = (scale,) * 3
    for o in arm.children:
        if o.type == "MESH":
            o.pass_index = index
    return arm, root


def grip(arm):
    """Palm point of the right hand in world space."""
    pb = arm.pose.bones
    pts = [pb[b].head for b in ("Bip01 R Finger1", "Bip01 R Finger2") if b in pb] or [pb["Bip01 R Hand"].head]
    p = sum(pts, mathutils.Vector()) / len(pts)
    return arm.matrix_world @ p


def bake_object(sc, ob, dims, arm, root, states, times, yaw_off, index, fps, setdown=0.0):
    """Object world positions per frame from its state timeline: hand (hangs from the palm, never below ground; moved
    forward by up to `setdown` m as the hand goes down, SET_HI / SET_LO), drop (frozen at release, falls to the
    ground), grasp (blends into the hand over 0.25 s), ground_at_release."""
    n = len(times)
    hand = np.zeros((n, 3)); yaw = np.zeros(n); side = np.zeros((n, 2))
    for k in range(n):
        sc.frame_set(k + 1)
        g = grip(arm); hand[k] = g
        yaw[k] = root.rotation_euler.z - math.pi / 2
        pel = arm.matrix_world @ arm.pose.bones["Bip01 Pelvis"].head
        v = np.array([g.x - pel.x, g.y - pel.y]); nv = np.linalg.norm(v)
        side[k] = v / nv if nv > 1e-6 else 0
    hang = hand.copy(); hang[:, :2] += side * 0.06; hang[:, 2] = np.maximum(0.0, hand[:, 2] - dims[2])
    if setdown:
        asc = root.scale[2]
        w = np.clip((SET_HI * asc - hand[:, 2]) / ((SET_HI - SET_LO) * asc), 0.0, 1.0)
        hang[:, :2] += setdown * w[:, None] * np.c_[np.cos(yaw), np.sin(yaw)]
    st = sorted(states, key=lambda s: s["t0"])

    def state_at(t):
        return [s for s in st if s["t0"] <= t][-1] if any(s["t0"] <= t for s in st) else st[0]

    def at_time(arr, t):  # frame k + 1 is clip time k / fps (the scenario's rate, not Blender's)
        k = min(n - 1, max(0, int(round(t * fps))))
        return arr[k]

    rest = None
    if any(s["state"] in ("drop", "ground_at_release") for s in st):
        tr = next(s["t0"] for s in st if s["state"] == "drop") if any(s["state"] == "drop" for s in st) else \
            next(s["t0"] for s in st if s["state"] == "grasp")
        rest = at_time(hang, tr).copy(); rest_yaw = at_time(yaw, tr)
    loc = np.zeros((n, 3)); rot = np.zeros(n); at_rest = np.zeros(n, bool)
    for k, t in enumerate(times):
        s = state_at(t)
        if s["state"] == "hand":
            loc[k], rot[k] = hang[k], yaw[k]
        elif s["state"] == "drop":
            dt = t - s["t0"]; z0 = rest[2]
            loc[k] = (rest[0], rest[1], max(0.0, z0 - 0.5 * G * dt * dt)); rot[k] = rest_yaw
            at_rest[k] = loc[k][2] <= 0.0
        elif s["state"] == "ground_at_release":
            loc[k] = (rest[0], rest[1], 0.0); rot[k] = rest_yaw; at_rest[k] = True
        elif s["state"] == "stow":  # loaded into the car: out of sight below the ground plane
            loc[k] = (hang[k][0], hang[k][1], -50.0); rot[k] = yaw[k]
        elif s["state"] == "grasp":
            w = min(1.0, (t - s["t0"]) / 0.25)
            base = np.array([rest[0], rest[1], 0.0])
            loc[k] = (1 - w) * base + w * hang[k]; rot[k] = (1 - w) * rest_yaw + w * yaw[k]
    frames = np.arange(1, n + 1, dtype=float)
    act = bpy.data.actions.new(ob.name + "_obj")
    for c in range(3):
        motion._write(act, "location", c, frames, loc[:, c])
    motion._write(act, "rotation_euler", 2, frames, rot + math.radians(yaw_off))
    ob.animation_data_create(); ob.animation_data.action = act
    ob.pass_index = index
    return loc, at_rest


def main():
    sc_path, out, opt = args()
    S = json.loads(sc_path.read_text()); gr, m = S["group"], S["member"]
    out.mkdir(parents=True, exist_ok=True)
    cal = group_calib(gr)  # the camera the scenario was sampled with
    fps = gr["render_fps"]
    n = int(math.floor(gr["duration"] * fps + 1e-6))
    sc, lamp_loc = setup(gr, cal, fps, opt["samples"], out)
    sc.frame_start, sc.frame_end = 1, n
    A = gr["actor"]
    arm, root = make_actor("actor", A["avatar"], A["gender"], A["scale"], 1)
    times = motion.bake(arm, root, m["segments"], m["root_keys"], A["gender"], fps, n, "actor")
    ob, dims = load_object(gr["object"]["asset"], gr["object"]["colour"], gr["object"].get("scale", 1.0))
    oloc, at_rest = bake_object(sc, ob, dims, arm, root, m["object_states"], times, gr["object"]["yaw"], 2, fps,
                                gr["object"].get("setdown", 0.0))
    walkers = [(arm, root, A["scale"])]
    for i, D in enumerate(gr["distractors"]):
        darm, droot = make_actor(f"dis{i}", D["avatar"], D["gender"], D["scale"], 3)
        segs = [dict(anim=D["walk"], t0=-120.0, t1=1e9, mode="loop")]
        motion.bake(darm, droot, segs, walker_keys(D["path"], D["t0"], D["speed"]), D["gender"], fps, n, f"dis{i}")
        if D.get("carry"):
            dob, ddims = load_object(D["carry"])
            bake_object(sc, dob, ddims, darm, droot, [dict(state="hand", t0=-1e9)], times, 0.0, 4, fps)
        walkers.append((darm, droot, D["scale"]))
    # per-frame screen extents (people as 1 m x 1 m x 2.1 m boxes, objects as their bbox, sun shadows on the ground,
    # the ground within SKY_R of each walker and of the object: the sky light's shadow, D39)
    ring = [(SKY_R * math.cos(a), SKY_R * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)]
    camx = Camera(cal, gr["render_size"]); W, H = gr["render_size"]
    L = gr["light"]
    sun = None
    if L["sun"]:
        az, el = math.radians(L["sun_az"]), math.radians(L["sun_el"])
        sun = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    meta = []
    lo_f, hi_f = (opt["frames"] or [1, n])
    for k in range(n):
        sc.frame_set(k + 1)
        pts = []
        heads = []
        for a_, r_, s_ in walkers:
            c = np.array(r_.matrix_world.translation)
            h = 2.1 * s_ * 1.0
            for dx in (-0.5, 0.5):
                for dy in (-0.5, 0.5):
                    for z in (0.0, h):
                        pts.append([c[0] + dx, c[1] + dy, z])
            pts += [[c[0] + dx * s_, c[1] + dy * s_, 0.0] for dx, dy in ring]
            hd = a_.matrix_world @ a_.pose.bones["Bip01 Head"].head
            heads.append([hd.x, hd.y, hd.z])
        o = oloc[k]
        for dx in (-dims[0], dims[0]):
            for dy in (-dims[1], dims[1]):
                for z in (0.0, dims[2]):
                    pts.append([o[0] + dx / 2, o[1] + dy / 2, o[2] + z])
        if o[2] > -1:  # (not stowed in the car)
            pts += [[o[0] + dx, o[1] + dy, 0.0] for dx, dy in ring]
        P = np.array(pts)
        if sun is not None:  # shadows on the ground along the sun direction
            P = np.vstack([P, P - (P[:, 2:3] / sun[2]) * sun[None]])
        if lamp_loc is not None:  # shadows cast by the night lamp (points below it)
            Q = P[(P[:, 2] > 0) & (P[:, 2] < lamp_loc[2] - 0.2)]
            P = np.vstack([P, lamp_loc + (Q - lamp_loc) * (lamp_loc[2] / (lamp_loc[2] - Q[:, 2:3]))])
        uv, z = camx.project(P)
        uv = uv[z > 0.1]
        rec = dict(frame=k + 1, t=round(float(times[k]), 4), rendered=False)
        hp, hz = camx.project(np.array(heads))
        neck = camx.project(np.array(heads) - [0, 0, 0.25])[0]
        rec["heads"] = [[float(u), float(v), float(abs(v - nv[1]))] for (u, v), nv, zz in zip(hp, neck, hz) if zz > 0.1]
        rec["obj_at_rest"] = bool(at_rest[k])
        if len(uv):
            x0, y0 = np.clip(uv.min(0) - 12, 0, [W, H]); x1, y1 = np.clip(uv.max(0) + 12, 0, [W, H])
            if x1 - x0 > 2 and y1 - y0 > 2 and lo_f <= k + 1 <= hi_f:
                sc.render.border_min_x, sc.render.border_max_x = x0 / W, x1 / W
                sc.render.border_min_y, sc.render.border_max_y = 1 - y1 / H, 1 - y0 / H
                bpy.ops.render.render(write_still=False)
                rec.update(rendered=True, border=[float(x0), float(y0), float(x1), float(y1)])
        meta.append(rec)
    (out / "meta.json").write_text(json.dumps(dict(n=n, fps=fps, size=[W, H], frames=meta)))
    print("RENDERED", sum(r["rendered"] for r in meta), "of", n, "frames ->", out)


if __name__ == "__main__":
    main()
