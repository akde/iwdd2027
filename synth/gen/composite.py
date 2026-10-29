"""Passes + plate -> clean mp4 (gen/build_scene.py output; all members of a group share one look).
python synth/gen/composite.py SCENARIO.json RENDER_DIR OUT.mp4
Plate frames: the last frame with PTS <= plate_start + k / fps (PTS-based, never frame index), resized to the
render size. Per frame (linear light): plate * shadow-catcher pass, then the foreground alpha-over.
Foreground look (shared by the group, stored in RENDER_DIR/../look.json by the first member processed, i.e. the
positive): actor gain so that actor / surrounding-plate luminance matches a ratio drawn from real people
(synth/data/person_ratio.npy; person_ratio_night.npy for night or dim plates), lowered where the render cannot show it
without blowing the actor's highlights (make_look, D39); the object and distractor objects get their own exposure
gain from the render's light and the plate's ground at the rest point (object_exposure, D44; the ground's albedo read
from the view on overcast plates, the camera's own illuminator on IR plates, D45); grey if the plate is IR, Gaussian
blur, plate-matched noise, face blur on heads. The shadow pass continues beyond the region render's border and fades
out (extend_shadow, D39); the strip a region render leaves inside its right border is capped (right_strip, D44).
Still parked vehicles are 3D solids fitted to their masks (fit_vehicle, D45) and hide a walker's pixels where they lie
between the camera and the walker along the pixel's ray (occluded, D45; short isolated hides dropped, occ_block).
Black bars (outside the plate's content box, plate_content.py) are outside the camera image: nothing rendered shows
there (the output is the plate, byte for byte), and the look's noise and gain come from the content only.
Plate structures (gen/structures.py, D34): groups sampled with structure masks (gr['structures']) get no synthetic
shadow on standing-structure or front-layer pixels (the shadow catcher is the ground plane); phase A, so structures
do not yet hide avatars (the sampler keeps walkers and the object off them).
Also writes RENDER_DIR/masks.npz: per frame object (index 2) and actor (index 1) pixel counts (inside the camera
image) and boxes, and obj_apart_px: visible object pixels farther than qc.apart_radius(actor box height) from the
actor's pixels (seen apart from the actor; qc.object_apart_s, D32)."""
import sys, json, math, pathlib
from fractions import Fraction
import av, cv2, numpy as np

from camera import Camera, group_calib
from tracks import track, walker_keys
import qc, scenario, structures

SYN = pathlib.Path(__file__).resolve().parents[1]


def plate_path(plate_id):
    import pandas as pd
    P = pd.read_csv(SYN / "data/plates.csv")
    return P.path[P.plate_id == plate_id].iloc[0]


def content_px(cal, size):
    """The camera image (calib content box) at size (W, H) in whole pixels, rounded inwards: x0, y0, x1, y1."""
    x0, y0, x1, y1 = Camera(cal, size).box
    return math.ceil(x0 - 1e-6), math.ceil(y0 - 1e-6), math.floor(x1 + 1e-6), math.floor(y1 + 1e-6)


def srgb2lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin2srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def plate_frames(path, t0, times, size):
    """RGB uint8 frames at PTS times t0 + times (last frame with pts <= t), resized to size (W, H)."""
    out, prev, p0, i = [], None, None, 0
    targets = [t0 + t for t in times]
    with av.open(str(path)) as c:
        s = c.streams.video[0]; s.thread_type = "AUTO"
        first = next(c.decode(s)); p0 = first.pts
        c.seek(max(0, int((t0 - 2.0) / s.time_base) + p0), stream=s, backward=True)
        for f in c.decode(s):
            if f.pts is None or (prev is not None and f.pts <= prev[0].pts):
                continue
            t = float((f.pts - p0) * s.time_base)
            while i < len(targets) and targets[i] < t and prev is not None:
                out.append(prev[1]); i += 1
            if i == len(targets):
                break
            if t <= targets[-1] + 1.0:
                prev = (f, f.reformat(width=size[0], height=size[1], format="rgb24", interpolation="AREA").to_ndarray())
        while i < len(targets):
            out.append(prev[1]); i += 1
    return out


def noise_sigma(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    r = g - cv2.GaussianBlur(g, (0, 0), 1.5)
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


def lum(x):
    return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]


SHADOW_EXT = 48  # px over which the shadow fades out beyond the region border (D39)


def extend_shadow(sh, box, ext=SHADOW_EXT):
    """Shadow pass sh known inside box (x0, y0, x1, y1, half-open), 1 outside. The region border cuts the sky
    light's shadow, which reaches about 2 m from a person (D39): outside the box each pixel takes the darkening
    1 - sh of the nearest box pixel (so the pass stays continuous at the box edge), faded to 0 over ext px
    (smoothstep). Inside the box sh is unchanged (the contact shadow)."""
    x0, y0, x1, y1 = box
    H, W = sh.shape
    if x1 - x0 < 1 or y1 - y0 < 1:
        return np.ones_like(sh)
    X0, Y0, X1, Y1 = max(0, x0 - ext), max(0, y0 - ext), min(W, x1 + ext), min(H, y1 + ext)
    xs, ys = np.arange(X0, X1), np.arange(Y0, Y1)
    cx, cy = np.clip(xs, x0, x1 - 1), np.clip(ys, y0, y1 - 1)
    r = np.minimum(1.0, np.hypot((xs - cx)[None, :], (ys - cy)[:, None]) / ext)
    out = np.ones_like(sh)
    out[Y0:Y1, X0:X1] = 1 - (1 - sh[np.ix_(cy, cx)]) * (1 - r * r * (3 - 2 * r))
    return out


STRIP_R = 16  # px: band inside the kept box's right edge (border - 2 px ring) where the strip is capped (D44)
STRIP_MAX = 0.08  # the strip's rise stays below this (pilot max 0.07): capped in full; from 2 x STRIP_MAX on not


def right_strip(sh, x0, y0, x1, y1, alpha=None, edge=False):
    """The shadow pass of a region render carries a soft dark strip 2-20 px inside its RIGHT border in about 3% of
    the frames (peak 0.03-0.07 darkening 5-9 px in, flickering; left, top and bottom are clean; D44: an edge effect of
    the denoised shadow catcher at the region's right side). (x0, y0, x1, y1): the kept box (the border minus its 2 px
    ring, half-open). Over its last STRIP_R columns the darkening 1 - sh may not grow towards the border: each row
    keeps the running minimum from the column before the band outwards, so a shadow that fades towards the border
    (every shadow the region padding holds) is unchanged and the strip is cut down to the level inside it; foreground
    pixels (alpha > 0, grown by 1 px: the pass sees no ground under them) set no limit and restart it; a row whose
    darkening rises by more than the strip does (STRIP_MAX) crosses a real shadow (a sun or lamp shadow's edge): the
    cap fades out linearly up to a rise of 2 x STRIP_MAX, where the row keeps its pass. The rows' edge pixels are what
    extend_shadow carries outwards, so this also removes the streaks it drew from the strip beyond the border. edge: the
    border is the image's right edge, where the region can cut a walker leaving the image (its contact shadow may grow
    towards the edge): with foreground (alpha > 0) within 2 x STRIP_R px of it the pass is kept."""
    a = max(x0 + 1, x1 - STRIP_R)
    if x1 - a < 2 or y1 - y0 < 1 or (edge and alpha is not None and (alpha[:, max(0, x1 - 2 * STRIP_R):] > 0).any()):
        return sh
    d = 1 - sh[y0:y1, a - 1:x1]
    fg = (cv2.dilate((alpha[y0:y1, a - 1:x1] > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
          if alpha is not None else np.zeros(d.shape, bool))  # the pass sees no ground under the foreground (about 1)
    c, cap = d.copy(), np.full(len(d), np.inf)
    for j in range(d.shape[1]):  # running minimum outwards, restarted after foreground pixels
        cap = np.where(fg[:, j], np.inf, np.minimum(cap, d[:, j]))
        c[:, j] = np.minimum(d[:, j], cap)
    w = np.clip((2 * STRIP_MAX - (d - c).max(1, keepdims=True)) / STRIP_MAX, 0, 1)  # a larger rise: a real shadow
    c = d - w * (d - c)
    sh = sh.copy(); sh[y0:y1, a:x1] = 1 - c[:, 1:]
    return sh


def load_pass(rd, k, border=None, shadow=True):
    """Passes of frame k. Outside the rendered border (region render) every pass is 0: alpha and index stay 0 there,
    the shadow pass continues the border's shadow and fades to 1 (unshadowed; extend_shadow) after the strip inside
    its right border is capped (right_strip). shadow=False: the shadow pass is not read (None)."""
    f = rd / f"rgba_{k:04d}.png"
    if not f.exists():
        return None
    rgba = cv2.cvtColor(cv2.imread(str(f), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA).astype(np.float32) / 255
    sh = None
    if shadow:
        sh = cv2.imread(str(rd / f"shadow_{k:04d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535
        sh = srgb2lin(sh if sh.ndim == 2 else sh[..., 0])  # the file output node saves it view-transformed (sRGB)
    idx = np.rint(cv2.imread(str(rd / f"index_{k:04d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 25.5)
    idx = idx if idx.ndim == 2 else idx[..., 0]
    if border is not None:  # the region's outermost pixels are partly computed (dark lines): drop a 2 px ring;
        x0, y0, x1, y1 = (int(round(v)) for v in border)  # the border is padded beyond any content (build_scene)
        inner = np.zeros(idx.shape, bool); inner[y0 + 2:y1 - 2, x0 + 2:x1 - 2] = True
        kept = (max(0, x0 + 2), max(0, y0 + 2), min(idx.shape[1], x1 - 2), min(idx.shape[0], y1 - 2))
        if shadow:
            sh = extend_shadow(right_strip(sh, *kept, rgba[..., 3], x1 >= sh.shape[1]), kept)
        rgba[~inner] = 0.0; idx = np.where(inner, idx, 0)
    return rgba, sh, idx


GAIN_MIN, GAIN_MAX = 0.05, 24.0  # gain clamp (8 before D39; the highlight bound limits the day groups)
CLIP_Q = 98  # after its gain this percentile of the actor's (and of the object's) brightest channel stays <= 1 (D39)
GROUND_ALBEDO = 0.2  # albedo of a typical walkable ground (asphalt 0.05-0.2, concrete 0.25-0.4; D44)
GROUND_RANGE = (0.1, 0.35)  # plausible ground albedos: the local estimate stays inside (D45)
MIN_WALK = 500  # walkable pixels needed for the local ground albedo estimate
AVATAR_ALBEDO = 0.18  # the avatars' albedo: actor / the sky's horizontal white radiance median 0.091 in the 24 overcast
# pilot positives (D45), an upright surface receiving half of it under the sky dome
GROUND_R = (0.1, 1.2)  # m beyond the rest box's corners: the annulus where the plate's ground luminance is read
MIN_GROUND = 20  # ground samples needed for an object exposure
LOOK_V = 45  # the look model (make_look) look.json carries; bump it when the look changes. run_batch --reuse --force
# keeps renders/<gid>/look.json: the positive remakes a look of another version, the negative takes the positive's (D46;
# before, batch_v3's D44 looks would have been kept and D45's object exposure never applied)
# the render's light as build_scene.setup makes it (synth/tests/test_compositor_d44.py checks these against its source)
SKY_RGB = {False: (0.55, 0.6, 0.7), True: (0.3, 0.3, 0.35)}  # world colour by day / at night
SKY_DAY, SKY_SUN, SKY_LAMP = 1.6, 0.7, 0.08  # world strength: overcast / night without a lamp, with sun, night lamp
SUN_W, LAMP_W = 3.5, 1500.0  # sun strength (W/m2), night lamp power (W)


def render_light(gr, meta=None):
    """The render's light from the scenario, as build_scene.setup makes it: dict(sky=world strength, sky_rgb,
    sun=(az, el) or None, lamp=xyz or None (drawn from the group seed as there)). meta['light'] overrides keys for
    renders of an older build_scene (before D39 night without a lamp had sky 0.08: pilot 1, release v2)."""
    L = gr["light"]
    lamp = None
    if L.get("night") and L.get("lamp"):
        X = gr["path"]["X"]; r = np.random.default_rng(gr["seed"])
        lamp = (X[0] + r.uniform(-5, 5), X[1] + r.uniform(-5, 5), r.uniform(4, 6))
    sun = bool(L.get("sun"))
    out = dict(sky=SKY_LAMP if lamp is not None else (SKY_SUN if sun else SKY_DAY),
               sky_rgb=SKY_RGB[bool(L.get("night"))], sun=(L["sun_az"], L["sun_el"]) if sun else None, lamp=lamp)
    out.update((meta or {}).get("light", {}))
    return out


def white_radiance(lt, P, cam):
    """Luminance of a white Lambertian surface at ground point P under the render's light lt (render_light): sky
    (radiance colour x strength) and sun (strength x sin(elevation) / pi) on a horizontal plane, as the plate's ground
    sees them (the pilot's sun shadows match this: umbra 0.31-0.45 against 0.32-0.47 predicted, D44); the night lamp
    (point light: power x cos / (4 pi^2 d^2)) on the faces the camera sees, the top and the vertical faces, weighted by
    the camera's elevation seen from the object, averaged over the lamp's azimuth (a vertical face gets cos(lamp
    elevation) / pi): the lamp is a random point near the spot, not a light of the plate, so its direction must not set
    the object's exposure (D45; D44 took the vertical face turned to the camera, and a lamp behind the object dimmed
    it: pilot g99650000's white bag 9.5 -> 1.7 x the ground)."""
    E = float(lum(np.asarray(lt["sky_rgb"], float))) * lt["sky"]
    if lt["sun"] is not None:
        E += SUN_W * max(0.0, math.sin(math.radians(lt["sun"][1]))) / math.pi
    if lt["lamp"] is not None:
        Q = np.array([P[0], P[1], 0.25]); v = np.asarray(lt["lamp"], float) - Q; d = np.linalg.norm(v); u = v / d
        c = cam.C - Q; s = max(0.0, c[2] / np.linalg.norm(c))
        E += LAMP_W / (4 * math.pi ** 2 * d * d) * (s * max(0.0, u[2]) + (1 - s) * math.hypot(u[0], u[1]) / math.pi)
    return float(E)


def object_faces(gr, cam):
    """The faces of the object's rest box (scenario.group_rest_box, object_height) that the camera sees: [(unit normal,
    centre, area (m2), cosine to the camera)]."""
    c, th, h = scenario.group_rest_box(gr)
    O = gr["object"]
    ht = scenario.object_height(O["asset"], O.get("scale", 1.0))
    a1, a2 = np.array([math.cos(th), math.sin(th), 0.0]), np.array([-math.sin(th), math.cos(th), 0.0])
    c3 = np.array([c[0], c[1], 0.0])
    faces = [(np.array([0.0, 0.0, 1.0]), c3 + (0, 0, ht), 4 * h[0] * h[1])]
    for sg in (1, -1):
        faces += [(sg * a1, c3 + sg * h[0] * a1 + (0, 0, ht / 2), 2 * h[1] * ht),
                  (sg * a2, c3 + sg * h[1] * a2 + (0, 0, ht / 2), 2 * h[0] * ht)]
    out = []
    for n, P, A in faces:
        v = cam.C - P; cs = float(n @ v / np.linalg.norm(v))
        if cs > 1e-3 and A > 0:
            out.append((n, P, float(A), cs))
    return out


def face_irradiance(lt, n, P):
    """The render's irradiance on a face (normal n at P): the sky dome's share (1 + n_z) / 2, the sun (by its elevation:
    the top gets sin, a vertical face the azimuth mean cos / pi), the lamp (power x cos / (4 pi^2 d^2), its own
    direction: what the render did)."""
    E = float(lum(np.asarray(lt["sky_rgb"], float))) * lt["sky"] * (1 + n[2]) / 2
    if lt["sun"] is not None:
        el = math.radians(lt["sun"][1])
        E += SUN_W / math.pi * (n[2] * max(0.0, math.sin(el)) + math.hypot(n[0], n[1]) * math.cos(el) / math.pi)
    if lt["lamp"] is not None:
        v = np.asarray(lt["lamp"], float) - P; d = float(np.linalg.norm(v))
        E += LAMP_W * max(0.0, float(n @ v) / d) / (4 * math.pi ** 2 * d * d)
    return float(E)


def ir_exposure(gr, cam, lt, lg):
    """IR plates (D45): the plate's light is the camera's own illuminator, so it lights the faces the camera sees by
    their cosine to the camera whatever the render's lamp did. Its level comes from real people in the same light: an
    upright avatar (albedo AVATAR_ALBEDO, its side's mean cosine to the camera pi / 4 of an upright face's) shows the
    median real night person / ground ratio (person_ratio_night.npy), as the actor does, so the plate's white radiance
    on a face is lg x ratio / (AVATAR_ALBEDO x pi / 4) x cos / cos_up (cos_up: an upright face's cosine to the camera):
    an object as light as the avatar is as bright as the actor beside it. Gain = the projected-area mean of that over
    the
    object's visible faces / the mean of the render's irradiance on them (face_irradiance: the render's lamp direction
    and its sky): the object's mean brightness no longer depends on where the render's lamp stood (pilot g99650000's
    white bag 1.7 x the ground with the lamp behind it, g99020012's 4.9 with it in front). -> (gain, render
    irradiance)"""
    faces = object_faces(gr, cam)
    if not faces:
        return None
    c = scenario.group_rest_box(gr)[0]
    v = cam.C - np.array([c[0], c[1], 0.0]); cos_up = max(math.hypot(v[0], v[1]) / float(np.linalg.norm(v)), 0.05)
    up = lg * float(np.median(np.load(SYN / "data/person_ratio_night.npy"))) / (AVATAR_ALBEDO * math.pi / 4)
    a = np.array([A * cs for n, P, A, cs in faces])
    Wp = np.array([up * cs / cos_up for n, P, A, cs in faces])
    Er = np.array([face_irradiance(lt, n, P) for n, P, A, cs in faces])
    E = float(a @ Er / a.sum())
    return float(a @ Wp / a.sum()) / max(E, 1e-9), E


def ground_albedo(gr, plate, cam, box, lg, vlab=None, solid=None):
    """The plate's ground albedo at the rest point (D45). Overcast day (uniform light, so luminance differences on the
    ground are albedo): GROUND_ALBEDO x the rest point's ground luminance lg / the median ground luminance of the
    view's walkable area (calib walkable polygon inside the camera image, off vehicles and structures), clipped to
    GROUND_RANGE: light concrete in a view of darker ground 0.3-0.35, grass or dark asphalt 0.1. Sun, night and dim
    (IR) plates keep GROUND_ALBEDO: there the ground's luminance follows the light (sun and shade, lamp pools, the IR
    illuminator's fall-off). Before D45 every plate had 0.2: a dark bag on light concrete showed 0.58 x the ground
    (pilot g99650017; plausible 0.2-0.4). The frame median as the reference was dropped: dark vegetation around brick
    pavers (G336) read the pavers as 0.39."""
    L = gr["light"]
    if L.get("sun") or L.get("night") or L.get("dim"):
        return GROUND_ALBEDO
    poly = np.asarray(group_calib(gr).get("walkable", []), float)
    if len(poly) < 3:
        return GROUND_ALBEDO
    uv, z = cam.project(np.c_[poly, np.zeros(len(poly))])
    if not (z > 0.5).all():  # (a polygon reaching behind the camera: its projection is not the area)
        return GROUND_ALBEDO
    H, W = plate.shape[:2]; x0, y0, x1, y1 = box
    wm = np.zeros((H, W), np.uint8)
    cv2.fillPoly(wm, [np.round(uv).astype(np.int32)], 1)
    wm = wm.astype(bool); wm[:y0] = False; wm[y1:] = False; wm[:, :x0] = False; wm[:, x1:] = False
    if vlab is not None:
        wm &= vlab == 0
    if solid is not None:
        wm &= ~solid
    if wm.sum() < MIN_WALK:
        return GROUND_ALBEDO
    ref = float(np.median(lum(srgb2lin(plate[wm].astype(np.float32) / 255))))
    return float(np.clip(GROUND_ALBEDO * lg / max(ref, 1e-6), *GROUND_RANGE))


def object_exposure(gr, plate, box, meta=None, vlab=None, solid=None):
    """The object's exposure gain (D44, D45) = the plate's light over the render's at the rest point. The ground
    luminance lg is the median linear luminance of the plate (the reference frame) on an annulus GROUND_R m beyond the
    rest box's corners (scenario.group_rest_box), inside the camera image (box), off vehicle masks and structures.
    Visible light: (lg / ground_albedo) / white_radiance, so the object shows the ground's luminance x its effective
    albedo / the ground's albedo whatever the actor wears; IR plates: ir_exposure. -> (gain, lg, the render's white
    radiance or face irradiance, dict(ground_albedo, model)), or None without a rest point or with < MIN_GROUND ground
    samples (make_look then shares the actor's gain as before D44)."""
    if "path" not in gr or "object" not in gr:
        return None
    H, W = plate.shape[:2]
    cam = Camera(group_calib(gr), (W, H))
    c, th, h = scenario.group_rest_box(gr)
    rr = np.hypot(*h) + np.linspace(*GROUND_R, 12)
    an = np.linspace(0, 2 * math.pi, 64, endpoint=False)
    P = c + (rr[:, None, None] * np.stack([np.cos(an), np.sin(an)], 1)[None]).reshape(-1, 2)
    uv, z = cam.project(np.c_[P, np.zeros(len(P))])
    x0, y0, x1, y1 = box
    ok = (z > 0.5) & (uv[:, 0] >= x0) & (uv[:, 0] < x1) & (uv[:, 1] >= y0) & (uv[:, 1] < y1)
    u, v = np.unique(np.floor(uv[ok]).astype(int), axis=0).T if ok.any() else (np.zeros(0, int), np.zeros(0, int))
    keep = np.ones(len(u), bool)
    if vlab is not None and len(u):
        keep &= vlab[v, u] == 0
    if solid is not None and len(u):
        keep &= ~solid[v, u]
    if keep.sum() < MIN_GROUND:
        return None
    lg = float(np.median(lum(srgb2lin(plate[v[keep], u[keep]].astype(np.float32) / 255))))
    lt = render_light(gr, meta)
    if gr["light"].get("ir") and (gr["light"].get("night") or gr["light"].get("dim")):  # (a grey day camera: not IR)
        ex = ir_exposure(gr, cam, lt, lg)
        if ex is not None:
            return ex[0], lg, ex[1], dict(ground_albedo=None, model="ir")
    ga = ground_albedo(gr, plate, cam, box, lg, vlab, solid)
    E = white_radiance(lt, c, cam)
    return lg / (ga * E), lg, E, dict(ground_albedo=ga, model="visible")


def make_look(gr, m, rd, meta, plates, box=None, vlab=None, solid=None):
    """Group look from this (positive) member at t_ref = arrival at the spot (identical in all members). box =
    content_px (default: the full frame): noise and the gain ring come from the camera image only, not the bars.
    Actor gain = ratio x plate ring / actor (linear luminance), ratio drawn from real people in the same light, but at
    most the gain that keeps the CLIP_Q percentile of the actor's brightest channel (luminance if IR; at t_ref and in
    the member's last frame with the object at rest) <= 1: a draw the render cannot show without blowing its
    highlights is lowered to the brightest ratio it can show (D39). look['ratio'] is the ratio the gain gives,
    look['ratio_drawn'] the draw. Object gain look['obj_gain'] (the object and distractor objects, D44):
    object_exposure, lowered where it would blow the object's own CLIP_Q highlight. Before D44 actor and object shared
    one gain bounded by both highlights: dark clothing pushed a light box to white and the box's highlight then capped
    the actor (pilot g99650005 / 6). Without an object exposure (no rest point or no ground in view) they still share
    it, bounded by both (D39)."""
    rng = np.random.default_rng(gr["seed"] + 7)
    fps = gr["render_fps"]
    k = min(len(plates) - 1, max(0, int(round(m["events"]["t_arrive"] * fps))))
    p = load_pass(rd, k + 1, meta["frames"][k].get("border"))
    bx0, by0, bx1, by1 = box if box is not None else (0, 0, plates[k].shape[1], plates[k].shape[0])
    inside = np.zeros(plates[k].shape[:2], bool); inside[by0:by1, bx0:bx1] = True
    dark = gr["light"]["night"] or gr["light"].get("dim", False)
    ratios = np.load(SYN / f"data/{'person_ratio_night' if dark else 'person_ratio'}.npy")  # real people, same light
    look = dict(gain=1.0, ratio=float(rng.choice(ratios)), blur=float(rng.uniform(0.3, 0.9)),
                noise=noise_sigma(plates[k][by0:by1, bx0:bx1]) * float(rng.uniform(0.8, 1.2)), ir=bool(gr["light"]["ir"]),
                t_ref=k / fps)
    if p is None:
        return look
    rest = [j for j in range(len(meta["frames"]) - 1, k, -1)
            if meta["frames"][j].get("rendered") and meta["frames"][j].get("obj_at_rest")][:1]
    v = {1: [], 2: []}  # brightest channel of the actor's and the object's core pixels
    for j, q in [(k, p)] + [(j, load_pass(rd, j + 1, meta["frames"][j].get("border"))) for j in rest]:
        if q is not None:
            lin = srgb2lin(q[0][..., :3]); b = lum(lin) if look["ir"] else lin.max(2)
            for c in v:
                v[c].append(b[(q[2] == c) & (q[0][..., 3] > 0.9) & inside])
    hi = {c: float(np.percentile(np.concatenate(x), CLIP_Q)) for c, x in v.items() if sum(map(len, x))}
    ex = object_exposure(gr, plates[k], (bx0, by0, bx1, by1), meta, vlab, solid)
    if ex is not None:
        og = min(GAIN_MAX, 1 / max(hi[2], 1e-6)) if 2 in hi else GAIN_MAX
        look.update(obj_gain=float(np.clip(ex[0], GAIN_MIN, max(GAIN_MIN, og))), obj_exposure=ex[0], ground=ex[1],
                    white=ex[2], obj_highlight=hi.get(2), **ex[3])
    rgba, sh, idx = p
    a = (idx == 1) & (rgba[..., 3] > 0.9) & inside
    if a.sum() > 30:
        ys, xs = np.nonzero(a)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        hh, ww = y1 - y0, x1 - x0
        ring = np.zeros_like(a)
        ring[max(0, y0 - hh // 2):y1 + hh // 2, max(0, x0 - ww):x1 + ww] = True
        ring &= (rgba[..., 3] < 0.01) & inside
        if ring.sum() >= 30:  # (an empty ring would make the gain NaN)
            lp = float(lum(srgb2lin(plates[k].astype(np.float32) / 255))[ring].mean())
            la = max(float(lum(srgb2lin(rgba[..., :3]))[a].mean()), 1e-4)
            h = hi[1] if ex is not None else max(hi.values())  # the actor's own highlight; shared: both (D39)
            gain = float(np.clip(look["ratio"] * lp / la, GAIN_MIN, max(GAIN_MIN, min(GAIN_MAX, 1 / max(h, 1e-6)))))
            look.update(gain=gain, ratio=gain * la / max(lp, 1e-6), ratio_drawn=look["ratio"], highlight=h)
    return look


OCC_REACH = 0.55  # m x the walker's scale: its feet, knees and a carried object reach this far from its root (D44)


def near_row(cam, xy, reach):
    """Image rows of ground points xy [N,2] moved `reach` m towards the camera along its viewing direction: rows grow
    with nearness for a camera without roll, so this is the walker's nearest reach as a row (inf behind the camera)."""
    f = -cam.R[:2, 2]; n = float(np.hypot(*f)); f = f / n if n > 1e-9 else f * 0
    P = np.atleast_2d(np.asarray(xy, float)) - np.reshape(np.asarray(reach, float), (-1, 1)) * f
    uv, z = cam.project(np.c_[P, np.zeros(len(P))])
    return np.where(z > 0.1, uv[:, 1], np.inf)


def occ_rows(gr, m, cam, ts):
    """Ground position of every walker at times ts as image coordinates (occluded, D44): its root track (tracks) as
    the foot column u and, as the depth row, the row of the root moved OCC_REACH x its scale towards the camera; the
    object's: its carrier's while in the hand, the rest box's (scenario.group_rest_box: centre column, nearest corner's
    row) while on the ground, blended over the 0.25 s grasp (build_scene.bake_object). -> per frame dict(rows={code:
    (row, u)}, ground={code: ground point (x, y)}, dis=[(foot uv, head uv, carries)]), codes as owners (1 actor, 2
    object, 10 + i / 20 + i distractor i and its object). 'ground' is what the 3D decision uses (D45): the walker's
    root, the object's carrier's root in the hand and its rest box's centre on the ground (blended likewise)."""
    ts = np.asarray(ts, float)
    xy = track(m["root_keys"], ts)
    a, au = near_row(cam, xy, OCC_REACH * gr["actor"]["scale"]), cam.project(np.c_[xy, np.zeros(len(ts))])[0][:, 0]
    c, th, h = scenario.group_rest_box(gr)
    corners = c + (np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]) * h) @ scenario.rot(th).T
    rest = float(near_row(cam, corners, 0.0).max())
    ru = float(cam.project(np.array([[c[0], c[1], 0.0]]))[0][0, 0])
    st = sorted(m["object_states"], key=lambda s: s["t0"])
    o = np.empty((len(ts), 2)); og = np.empty((len(ts), 2)); ow = np.empty(len(ts))
    ra, rr = OCC_REACH * gr["actor"]["scale"], float(np.hypot(*h)) + 0.1  # walker reach; resting object's half diagonal
    for i, t in enumerate(ts):
        s = ([x for x in st if x["t0"] <= t] or st[:1])[-1]
        w = 0.0 if s["state"] in ("drop", "ground_at_release") else (
            min(1.0, (t - s["t0"]) / 0.25) if s["state"] == "grasp" else 1.0)
        o[i] = (rest, ru) if w == 0 else (a[i], au[i]) if w == 1 else (  # (rows may be inf: never hidden)
            (1 - w) * np.array([rest, ru]) + w * np.array([a[i], au[i]]))
        og[i] = (1 - w) * np.asarray(c, float) + w * xy[i]; ow[i] = w
    dis = []
    for D in gr.get("distractors", []):
        dxy = track(walker_keys(D["path"], D["t0"], D["speed"]), ts)
        foot, head, _ = cam.person_px(dxy, 1.75 * D["scale"])
        dis.append((near_row(cam, dxy, OCC_REACH * D["scale"]), foot, head, bool(D.get("carry")), dxy, D["scale"]))
    out = []
    for i in range(len(ts)):
        rows = {1: (float(a[i]), float(au[i])), 2: (float(o[i, 0]), float(o[i, 1]))}
        ground = {1: (float(xy[i, 0]), float(xy[i, 1])), 2: (float(og[i, 0]), float(og[i, 1]))}
        reach = {1: ra, 2: max(ra, rr) if 0 < ow[i] < 1 else (ra if ow[i] == 1 else rr)}
        for j, (r, f, hd, cy, dxy, sc) in enumerate(dis):
            rows[10 + j] = rows[20 + j] = (float(r[i]), float(f[i][0]))
            ground[10 + j] = ground[20 + j] = (float(dxy[i, 0]), float(dxy[i, 1]))
            reach[10 + j] = reach[20 + j] = OCC_REACH * sc
        out.append(dict(rows=rows, ground=ground, reach=reach, dis=[(f[i], hd[i], cy) for r, f, hd, cy, _, _ in dis]))
    return out


def seg_dist(P, a, b):
    """Distance of points P [N,2] to the segment a-b."""
    a, b = np.asarray(a, float), np.asarray(b, float); d = b - a
    t = np.clip(((P - a) @ d) / max(float(d @ d), 1e-9), 0, 1)
    return np.hypot(*(P - a - t[:, None] * d).T)


def owners(idx, alpha, occ=None):
    """Who each pixel near the foreground belongs to (D44): code 1 actor, 2 object, 10 + i distractor i (index 3, the
    nearest projected body of occ['dis']), 20 + i its object (index 4, the nearest carrying distractor); fringe pixels
    (alpha > 0, index 0) and the rest of the window take the code of the nearest indexed pixel. -> ((y0, y1, x0, x1)
    window around the foreground, codes in it), or None without foreground."""
    fg = (alpha > 0.004) | (idx > 0)
    if not fg.any():
        return None
    ys, xs = np.nonzero(fg); H, W = idx.shape
    y0, y1, x0, x1 = max(0, ys.min() - 2), min(H, ys.max() + 3), max(0, xs.min() - 2), min(W, xs.max() + 3)
    code = idx[y0:y1, x0:x1].astype(np.int32)
    ds = (occ or {}).get("dis", [])
    for c, base in ((3, 10), (4, 20)):
        m = code == c
        if not m.any():
            continue
        cand = [i for i, d in enumerate(ds) if c == 3 or d[2]] or list(range(len(ds))) or [0]
        if len(cand) == 1:
            code[m] = base + cand[0]; continue
        py, px = np.nonzero(m)
        P = np.c_[px + x0, py + y0].astype(float)
        code[m] = base + np.array(cand)[np.stack([seg_dist(P, ds[i][0], ds[i][1]) for i in cand], 1).argmin(1)]
    if (code == 0).any() and (code > 0).any():
        src = (code == 0).astype(np.uint8)
        _, lab = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
        lut = np.zeros(lab.max() + 1, np.int32); lut[lab[src == 0]] = code[src == 0]
        code = lut[lab]
    return (y0, y1, x0, x1), code


# ---------------------------------------------------------------- still parked vehicles as 3D solids (D45)
# per COCO class (plate_vehicles.py): length, width, height ranges (m) and the cabin (top length and width as fractions
# of the footprint, body height as a fraction of H; 1, 1, 1: a box); other classes are taken as cars
VEH_DIMS = {2: ((3.6, 5.4), (1.6, 2.05), (1.35, 1.95), (0.55, 0.85, 0.6)),  # car (SUVs up to 1.95 m)
            7: ((4.6, 12.0), (1.8, 2.6), (1.7, 3.8), (1.0, 1.0, 1.0)),  # truck (pickups to box trucks)
            5: ((8.0, 14.0), (2.3, 2.6), (2.7, 3.6), (1.0, 1.0, 1.0)),  # bus
            3: ((1.6, 2.4), (0.6, 0.95), (1.0, 1.5), (1.0, 1.0, 1.0)),  # motorcycle
            1: ((1.5, 2.0), (0.5, 0.8), (0.9, 1.3), (1.0, 1.0, 1.0))}  # bicycle
VEH_YAWS = 24  # yaw starts over 180 deg for the fit (without a usable bottom contour)
VEH_MIN_IOU = 0.7  # a fit below this silhouette IoU gives no depth (D44 rule for that vehicle; car-mode cars >= 0.78)
VEH_GROW = (0.1, 0.25, 0.5, 1.0)  # m: solids grown for mask pixels whose ray misses the fitted one
VEH_LOOSE, VEH_SOFT = (0.6, 1.4), 0.3  # the class ranges are soft (cost VEH_SOFT per relative excess) within these
# factors: the plate calibrations distort sizes locally (G336: a pickup's wheelbase comes out 2.6 m), and the solid must
# match the mask in the calibrated world the walkers are rendered in


def vehicle_solid(p, cls):
    """Vertices [12, 3] of a vehicle's solid: the ground footprint (centre cx, cy, yaw th, length L, width W) up to the
    body height, then the cabin's top at H (cars; a box for the other classes). p = (cx, cy, th, L, W, H)."""
    cx, cy, th, L, W, H = p
    fl, fw, fb = VEH_DIMS.get(cls, VEH_DIMS[2])[3]
    ca, sa = math.cos(th), math.sin(th)
    out = []
    for l, w, z in ((L, W, 0.0), (L, W, fb * H), (fl * L, fw * W, H)):
        for sl, sw in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
            x, y = sl * l / 2, sw * w / 2
            out.append((cx + ca * x - sa * y, cy + sa * x + ca * y, z))
    return np.array(out)


def solid_planes(p, cls):
    """Outward face planes of vehicle_solid (convex): n [F, 3], d [F]; a point X is inside where n @ X + d <= 0 for
    every face. Faces: bottom, top, the 4 body sides and the 4 cabin sides (degenerate ones of a box are dropped)."""
    V = vehicle_solid(p, cls); c = V.mean(0)
    faces = [(0, 1, 2, 3), (8, 9, 10, 11)] + [(i, (i + 1) % 4, 4 + (i + 1) % 4, 4 + i) for i in range(4)] + \
            [(4 + i, 4 + (i + 1) % 4, 8 + (i + 1) % 4, 8 + i) for i in range(4)]
    ns, ds = [], []
    for f in faces:
        P = V[list(f)]
        n = np.zeros(3)
        for a, b in zip(P, np.roll(P, -1, 0)):  # Newell's normal: robust for any planar polygon, 0 if degenerate
            n += ((a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1]))
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n); d = -float(n @ P.mean(0))
        if n @ c + d > 0:
            n, d = -n, -d
        ns.append(n); ds.append(d)
    return np.array(ns), np.array(ds)


def ray_dirs(cam, u, v):
    """Unit world directions [N, 3] of the camera rays through pixel coordinates (u, v)."""
    d = np.stack([(np.asarray(u, float) - cam.W / 2) / cam.f, -(np.asarray(v, float) - cam.H / 2) / cam.f,
                  -np.ones(np.size(u))], 1) @ cam.R.T
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def ray_interval(planes, C, D):
    """Ray parameters (t_in, t_out) where the rays C + t D [N] cross a convex solid (planes: solid_planes); t_in > t_out
    where they miss it."""
    n, d = planes
    num = n @ C + d  # [F]
    den = D @ n.T  # [N, F]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -num[None, :] / den
    t_in = np.where(den < -1e-12, t, -np.inf).max(1)
    t_out = np.where(den > 1e-12, t, np.inf).min(1)
    par = (np.abs(den) <= 1e-12) & (num[None, :] > 0)  # parallel to a face and outside it
    t_in = np.where(par.any(1), np.inf, np.maximum(t_in, 0.0))
    return t_in, t_out


def nelder_mead(f, x0, step, iters, tol):
    """Minimise f from x0 (simplex steps `step`): plain Nelder-Mead, deterministic."""
    n = len(x0)
    X = np.vstack([x0] + [x0 + np.eye(n)[i] * step[i] for i in range(n)]); F = np.array([f(x) for x in X])
    for _ in range(iters):
        o = np.argsort(F, kind="stable"); X, F = X[o], F[o]
        if F[-1] - F[0] < tol:
            break
        c = X[:-1].mean(0); xr = c + (c - X[-1]); fr = f(xr)
        if fr < F[0]:
            xe = c + 2 * (c - X[-1]); fe = f(xe)
            X[-1], F[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < F[-2]:
            X[-1], F[-1] = xr, fr
        else:
            xc = c + 0.5 * (X[-1] - c); fc = f(xc)
            if fc < F[-1]:
                X[-1], F[-1] = xc, fc
            else:
                X[1:] = X[0] + 0.5 * (X[1:] - X[0]); F[1:] = [f(x) for x in X[1:]]
    i = int(np.argmin(F))
    return X[i], float(F[i])


def lower_envelope(u, b):
    """The lowest-in-the-image convex hull of the points (u, b) (rows grow downwards), read at u: a ground footprint's
    near boundary projects to a polyline that is convex downwards; the wheels touch it, sills and bumpers lie above."""
    pts = []
    for p in zip(np.asarray(u, float), np.asarray(b, float)):
        while len(pts) >= 2 and ((pts[-1][0] - pts[-2][0]) * (p[1] - pts[-2][1])
                                 - (pts[-1][1] - pts[-2][1]) * (p[0] - pts[-2][0])) >= 0:
            pts.pop()
        pts.append(p)
    P = np.array(pts)
    return np.interp(u, P[:, 0], P[:, 1])


def lshape_yaw(Q, Cg):
    """Yaw (rad, [0, pi / 2)) of a rectangle whose near edges (those facing the camera at ground point Cg) pass through
    the ground points Q [N, 2] (the back-projected lower envelope of a vehicle's mask: wheels and corners): per yaw the
    two near edges are put through the nearest points (5th percentile) along each axis, and the yaw with the smallest
    trimmed (80%) mean distance of the points to the nearer edge wins. -> (yaw, that distance in m, near edges)."""
    best = None
    for th in np.radians(np.arange(0.0, 90.0, 1.0)):
        R = np.array([[math.cos(th), math.sin(th)], [-math.sin(th), math.cos(th)]])
        q, c = Q @ R.T, Cg @ R.T
        nx = np.percentile(q[:, 0], 5 if c[0] < q[:, 0].mean() else 95)
        ny = np.percentile(q[:, 1], 5 if c[1] < q[:, 1].mean() else 95)
        r = np.sort(np.minimum(np.abs(q[:, 0] - nx), np.abs(q[:, 1] - ny)))
        sc = float(r[:max(1, int(0.8 * len(r)))].mean())
        if best is None or sc < best[1] - 1e-9:
            best = (float(th), sc, (nx, ny))
    return best


def fit_vehicle(cam, vlab, vid, cls, contact_uv):
    """A vehicle's 3D solid from its mask (D45): footprint centre, yaw, length, width and height, so that through the
    camera (1) the solid's silhouette matches the mask (IoU; pixels of other vehicles, which may cover it, and the
    outside of the camera image do not count; the silhouette's area outside the fitting window does), (2) the
    projection of its footprint's near boundary matches the lower envelope of the mask's bottom contour (the wheels;
    columns cut by the image or another vehicle left out), (3) that envelope's ground points lie on the footprint's
    edges (none outside it, the nearest decile on an edge). The sizes are soft within the class's ranges (VEH_DIMS,
    VEH_SOFT) and free within VEH_LOOSE of them: the plates' calibrations distort sizes locally, and the solid must
    match the mask in the calibrated world the walkers are rendered in. Started at the yaw of the envelope's L-shape
    (lshape_yaw) and 90 deg from it (the yaw then stays within 6 deg), else from VEH_YAWS yaws; the best three starts
    refined. -> dict(p=(cx, cy, th, L, W, H), cls, iou, cost)."""
    cls = cls if cls in VEH_DIMS else 2
    Lr, Wr, Hr, _ = VEH_DIMS[cls]
    LIM = [(VEH_LOOSE[0] * a, VEH_LOOSE[1] * b) for a, b in (Lr, Wr, Hr)]
    m = vlab == vid
    ys, xs = np.nonzero(m)
    Hh, Ww = vlab.shape
    bx0, by0, bx1, by1 = math.ceil(cam.box[0] - 1e-6), math.ceil(cam.box[1] - 1e-6), math.floor(cam.box[2] + 1e-6), \
        math.floor(cam.box[3] + 1e-6)
    pw, ph = max(12, int(0.3 * (xs.max() - xs.min()))), max(12, int(0.3 * (ys.max() - ys.min())))
    x0, x1 = max(bx0, xs.min() - pw), min(bx1, xs.max() + pw + 1)
    y0, y1 = max(by0, ys.min() - ph), min(by1, ys.max() + ph + 1)
    win = m[y0:y1, x0:x1]
    care = (vlab[y0:y1, x0:x1] == 0) | win
    cols = np.flatnonzero(win.any(0))
    bot = win.shape[0] - 1 - np.argmax(win[::-1, cols], 0)
    ok = (cols + x0 > bx0 + 1) & (cols + x0 < bx1 - 2) & (bot + y0 < by1 - 2)
    below = vlab[np.minimum(bot + y0 + 2, Hh - 1), cols + x0]
    ok &= (below == 0) | (below == vid)
    bc = cols[ok]
    benv = lower_envelope(bc, bot[ok]) if len(bc) >= 3 else None
    mh = float(ys.max() - ys.min() + 1)
    canvas = np.zeros(win.shape, np.uint8)
    Q = None  # the envelope on the ground: the wheels' contacts and the corners lie on the footprint's near edges
    if benv is not None and len(bc) >= 10:
        Q = cam.ground(np.c_[bc + x0 + 0.5, benv + y0 + 0.5])
        Q = Q[np.isfinite(Q).all(1)]
        if len(Q) < 10 or np.ptp(Q, 0).max() < 1.0:
            Q = None

    img_rect = np.array([[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]], np.float32)

    def hull(q):
        """The solid's silhouette in the window, and its area inside the camera image but outside the window."""
        cx, cy, th, L, W, H = q
        dims = (np.clip(L, *LIM[0]), np.clip(W, *LIM[1]), np.clip(H, *LIM[2]))
        uv, z = cam.project(vehicle_solid((cx, cy, th, *dims), cls))
        if (z < 0.5).any():
            return None, 0.0
        canvas[:] = 0
        cv2.fillConvexPoly(canvas, cv2.convexHull(np.round((uv - (x0, y0)) * 4).astype(np.int32)), 1, cv2.LINE_8, 2)
        hm = canvas.astype(bool)
        a_img = cv2.intersectConvexConvex(cv2.convexHull(uv.astype(np.float32)), img_rect)[0]
        return hm, max(0.0, float(a_img) - float(hm.sum()))

    def cost(q):
        hm, out = hull(q)
        if hm is None:
            return 10.0
        pen = sum(2.0 * (max(0.0, a - v) + max(0.0, v - b)) for v, (a, b) in zip(q[3:], LIM))  # hard
        pen += sum(VEH_SOFT * (max(0.0, a - v) / a + max(0.0, v - b) / b) for v, (a, b) in zip(q[3:], (Lr, Wr, Hr)))
        c = 1.0 - float((hm & win).sum()) / max(float(((hm | win) & care).sum()) + out, 1.0) + pen
        if benv is not None:
            hb = np.where(hm[:, bc].any(0), hm.shape[0] - 1 - np.argmax(hm[::-1, bc], 0), np.nan)
            d = np.minimum(np.abs(hb - benv), 0.25 * mh)
            c += float(np.nanmean(d) if np.isfinite(d).any() else 0.25 * mh) / mh
        if Q is not None:  # those ground points lie on the footprint's edges: none outside (m beyond 0.15), and the
            # nearest decile within 0.1 m of an edge (sills and bumpers above the ground land further in)
            ca, sa = math.cos(q[2]), math.sin(q[2]); dq = Q - q[:2]
            x = np.abs(dq @ (ca, sa)) - np.clip(q[3], *LIM[0]) / 2
            y = np.abs(dq @ (-sa, ca)) - np.clip(q[4], *LIM[1]) / 2
            c += float(np.maximum(np.hypot(np.maximum(x, 0), np.maximum(y, 0)) - 0.15, 0).mean())
            c += max(0.0, float(np.percentile(-np.maximum(x, y), 10)) - 0.1)
        return c

    C0 = cam.ground(np.array([contact_uv], float))[0]
    fwd = C0 - cam.C[:2]; fwd /= max(np.linalg.norm(fwd), 1e-9)
    Lm, Wm, Hm = np.mean(Lr), np.mean(Wr), np.mean(Hr)
    th0 = None
    if Q is not None:
        th0, res, near = lshape_yaw(Q, cam.C[:2])
        if res > 0.15:
            th0 = None
    starts = []
    yaws = [th0, th0 + math.pi / 2] if th0 is not None else list(np.arange(VEH_YAWS) * math.pi / VEH_YAWS)
    for th in yaws:
        if th0 is not None:  # the near corner where the L found it, the footprint extending away from the camera
            R = np.array([[math.cos(th), math.sin(th)], [-math.sin(th), math.cos(th)]])
            q, cc = Q @ R.T, cam.C[:2] @ R.T
            sx = 1.0 if cc[0] < q[:, 0].mean() else -1.0; sy = 1.0 if cc[1] < q[:, 1].mean() else -1.0
            nx = np.percentile(q[:, 0], 5 if sx > 0 else 95); ny = np.percentile(q[:, 1], 5 if sy > 0 else 95)
            c = np.array([nx + sx * Lm / 2, ny + sy * Wm / 2]) @ R
        else:
            ext = 0.5 * (abs(Lm * (math.cos(th) * fwd[0] + math.sin(th) * fwd[1]))
                         + abs(Wm * (-math.sin(th) * fwd[0] + math.cos(th) * fwd[1])))
            c = C0 + ext * fwd
        q5, f = nelder_mead(lambda q5: cost(np.r_[q5[:2], th, q5[2:]]), np.array([c[0], c[1], Lm, Wm, Hm]),
                            [0.6, 0.6, 0.5, 0.15, 0.2], 150, 1e-4)
        starts.append((f, np.r_[q5[:2], th, q5[2:]]))
    starts.sort(key=lambda s: s[0])

    def cost6(q):  # the yaw free, but within 6 deg of the L's when it has one
        dy = abs((q[2] - starts_th + math.pi / 2) % math.pi - math.pi / 2)
        return cost(q) + (2.0 * max(0.0, dy - math.radians(6)) if th0 is not None else 0.0)
    best = None
    for f, q in starts[:3]:
        starts_th = q[2]
        r = nelder_mead(cost6, q, [0.3, 0.3, 0.06, 0.3, 0.1, 0.1], 300, 1e-5)
        if best is None or r[1] < best[1]:
            best = r
    cx, cy, th, L, W, H = best[0]
    p = (float(cx), float(cy), float(th % math.pi), float(np.clip(L, *LIM[0])), float(np.clip(W, *LIM[1])),
         float(np.clip(H, *LIM[2])))
    hm, out = hull(p)
    iou = float((hm & win).sum()) / max(float(((hm | win) & care).sum()) + out, 1.0) if hm is not None else 0.0
    return dict(p=p, cls=cls, iou=iou, cost=best[1])


_SOLIDS = {}


def plate_solids(plate_id, cam_calib, size, vlab, vehicles):
    """fit_vehicle for each still vehicle of a plate (vehicles: [(id, cls, contact_uv)]), cached per plate, camera and
    size in this process. -> {id: fit}"""
    key = (plate_id, json.dumps(cam_calib, sort_keys=True), tuple(size))
    if key not in _SOLIDS:
        cam = Camera(cam_calib, size)
        _SOLIDS[key] = {k: fit_vehicle(cam, vlab, k, cls, cuv) for k, cls, cuv in vehicles}
    return _SOLIDS[key]


def vehicle_depth(cam, vlab, solids):
    """Per pixel of each still vehicle's mask the camera ray's interval through its fitted solid (D45): t_in, t_out
    (metres along the unit ray; nan off the masks and on vehicles whose fit has IoU < VEH_MIN_IOU: masks cut by a
    wall or a tree in front of the vehicle, 23 of 322 still vehicles on the plan's and the pilots' plates). A mask pixel whose ray misses the solid (the silhouettes differ at
    the edges) takes its interval through the solid grown by VEH_GROW steps (footprint and height), so the intervals
    stay smooth over the mask; what still misses takes the nearest pixel's."""
    tin = np.full(vlab.shape, np.nan, np.float32); tout = np.full(vlab.shape, np.nan, np.float32)
    for k, f in solids.items():
        ys, xs = np.nonzero(vlab == k)
        if not len(ys) or f["iou"] < VEH_MIN_IOU:  # (no depth: occluded falls back to the D44 rule for it)
            continue
        D = ray_dirs(cam, xs + 0.5, ys + 0.5)
        a, b = ray_interval(solid_planes(f["p"], f["cls"]), cam.C, D)
        hit = a < b
        cx, cy, th, L, W, H = f["p"]
        for g in VEH_GROW:
            if hit.all():
                break
            a2, b2 = ray_interval(solid_planes((cx, cy, th, L + 2 * g, W + 2 * g, H + g), f["cls"]), cam.C, D[~hit])
            i = np.flatnonzero(~hit)[a2 < b2]
            a[i], b[i] = a2[a2 < b2], b2[a2 < b2]; hit[i] = True
        if not hit.any():
            continue
        if not hit.all():  # nearest hitting pixel (within the vehicle's mask box)
            y0, x0 = ys.min(), xs.min()
            src = np.ones((ys.max() - y0 + 1, xs.max() - x0 + 1), np.uint8)
            src[ys[hit] - y0, xs[hit] - x0] = 0
            _, lab = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
            ia = np.zeros(lab.max() + 1, np.int64); ia[lab[ys[hit] - y0, xs[hit] - x0]] = np.flatnonzero(hit)
            j = ia[lab[ys - y0, xs - x0]]
            a, b = a[j], b[j]
        tin[ys, xs], tout[ys, xs] = a, b
    return dict(cam=cam, tin=tin, tout=tout)


def still_vehicles(vlab, vcols, depth=None):
    """Still parked vehicles for occluded: (vlab, vcols: lowest mask row per column of each, per pixel the
    ground-contact row of the vehicle there (-1 off them), depth: vehicle_depth or None). With depth, occluded decides
    in 3D (D45); without (no camera: the fabricated vehicles of the D44 tests) by the D44 per-column rows."""
    vc = np.full(vlab.shape, -1, np.int32)
    for k, cb in vcols.items():
        m = vlab == k
        vc[m] = np.broadcast_to(np.asarray(cb, np.int32)[None, :], vlab.shape)[m]
    return vlab, vcols, vc, depth


def occluded(idx, alpha, veh, occ, own=None):
    """Foreground pixels hidden by still parked vehicles (plate_vehicles.py), geometric and per walker. veh:
    still_vehicles; occ: the frame's occ_rows; own: owners (computed if None). Every pixel has an owner (a walker, the
    object, a distractor or its object); the hidden area grows by 1 px under the same rule (no ghost outline).
    D45, with the vehicles' solids (still_vehicles with vehicle_depth): a pixel on a vehicle's mask is hidden where
    (a) its owner's upright axis (its ground point; the carried object goes with its carrier, a resting object stands at
    its rest box) lies along the pixel's ray past the middle of the ray's interval through the solid: a walker beside a
    corner whose ray only grazes it belongs to the nearer side, as nothing of it can be inside the vehicle;
    (b) the owner's reach (occ['reach'], OCC_REACH x scale around the axis, the D38/D41 envelopes) extends along the
    ray past the solid's far side: a walker in front of a long side whose box or arm reaches over the car is in front;
    (c) the vehicle's lowest mask row in the pixel's column is lower (nearer) than the owner's own ground row;
    and the pair is not blocked in this frame by the temporal filter (occ['block'], occ_block).
    D44 (vehicles without depth: fabricated ones without a camera, fits below VEH_MIN_IOU): a pixel is hidden where the vehicle's ground contact in its column is
    lower than its owner's depth row (occ['rows']: the ground point moved OCC_REACH x scale towards the camera), except
    where the vehicle spans the owner's own foot column with the owner in front of it there. That exemption switched
    off as a walker's foot column left a car's last column while its body still overlapped the car's end: 10-75 px
    vanished for 1-4 frames (E2: 54 events in 8 car-mode groups of the batch plan)."""
    hidden = np.zeros(idx.shape, bool)
    if veh is None or occ is None:
        return hidden
    vlab, vcols, vc = veh[:3]
    depth = veh[3] if len(veh) > 3 else None
    fg = (alpha > 0.004) | (idx > 0)
    if not (fg & (vc >= 0)).any():
        return hidden
    (y0, y1, x0, x1), code = own if own is not None else owners(idx, alpha, occ)
    v, lab = vc[y0:y1, x0:x1], vlab[y0:y1, x0:x1]
    d3 = depth is not None and "ground" in occ
    has = np.isfinite(depth["tin"][y0:y1, x0:x1]) if d3 else np.zeros(v.shape, bool)
    if d3:  # D45: in 3D, along each pixel's camera ray
        h = np.zeros(v.shape, bool)
        yy, xx = np.nonzero((v >= 0) & has)
        cc = code[yy, xx]
        cam = depth["cam"]
        Dh = ray_dirs(cam, xx + x0 + 0.5, yy + y0 + 0.5)[:, :2]
        ta = np.full(len(yy), -np.inf); tf = np.full(len(yy), -np.inf)  # the walker along each ray: axis, far reach
        grow = np.full(int(code.max()) + 1, np.inf)  # each owner's own ground row (no reach)
        for c, g in occ["ground"].items():
            sel = cc == c
            if c < len(grow):
                grow[c] = cam.project(np.array([[g[0], g[1], 0.0]]))[0][0, 1]
            if sel.any():
                rel = np.asarray(g, float) - cam.C[:2]; d2 = (Dh[sel] ** 2).sum(1)
                t = Dh[sel] @ rel / d2  # nearest approach to the walker's upright axis
                perp2 = np.maximum(rel @ rel - t * t * d2, 0.0)
                ta[sel] = t; tf[sel] = t + np.sqrt(np.maximum(occ["reach"][c] ** 2 - perp2, 0.0) / d2)
        tin, tout = depth["tin"][yy + y0, xx + x0], depth["tout"][yy + y0, xx + x0]
        h[yy, xx] = (ta > 0.5 * (tin + tout)) & (tf > tout) & (v[yy, xx] > grow[cc])
    if not d3 or ((v >= 0) & ~has).any():  # D44: per column, by depth rows (vehicles without depth)
        lut = np.full(max(30, int(code.max()) + 1), np.inf)
        for c, (r, u) in occ["rows"].items():
            if c < len(lut):
                lut[c] = r
        h4 = (v >= 0) & (v > lut[code])
        for c, (r, u) in occ["rows"].items():  # in front of a vehicle at its own position: never behind it
            ui = int(np.floor(u)) if np.isfinite(u) else -1
            if 0 <= ui < vc.shape[1] and (code == c).any():
                for k, cb in vcols.items():
                    if 0 <= cb[ui] <= r:
                        h4 &= ~((code == c) & (lab == k))
        h = np.where(has, h, h4) if d3 else h4
    for c, k in (occ.get("block", ()) if d3 else ()):  # pairs the temporal filter keeps visible here (occ_block)
        h &= ~((code == c) & (lab == k))
    h0 = (h & fg[y0:y1, x0:x1]).astype(np.uint8)
    hidden[y0:y1, x0:x1] = cv2.dilate(h0, np.ones((3, 3), np.uint8)).astype(bool) & h
    return hidden


OCC_MIN_S = 0.2  # s: shorter hide episodes of a walker-vehicle pair between visible stretches this long are dropped


def short_hides(h, nmin):
    """Frames of the hide episodes (runs of h > 0) shorter than nmin frames with at least nmin frames of h == 0 on
    both sides (an episode at the clip's start or end is kept)."""
    h = np.asarray(h) > 0; out = np.zeros(len(h), bool); n = len(h); i = 0
    while i < n:
        if not h[i]:
            i += 1; continue
        j = i
        while j < n and h[j]:
            j += 1
        if j - i < nmin and i >= nmin and j + nmin <= n and not h[i - nmin:i].any() and not h[j:j + nmin].any():
            out[i:j] = True
        i = j
    return out


def occ_block(occ, veh, rd, meta, cut=None):
    """Temporal filter per walker-vehicle pair (D45), a safety net under the 3D rule: a first pass over the index and
    alpha passes counts each pair's hidden pixels per frame (occluded); a hide episode shorter than OCC_MIN_S between
    visible stretches at least that long is a flicker of the model at a decision boundary (a vehicle hides a walker for
    longer than that when it really passes behind it), and the pair stays visible in it: occ[k]['block'] = {(owner
    code, vehicle id)}. cut: the member's divergence frame. The frames before it are judged on those frames alone (the
    cut counts as the clip's end), so a group's members, identical until they diverge, get the same decisions there.
    -> occ, the number of frames blocked."""
    if veh is None or veh[3] is None:
        return occ, 0
    n = len(occ); nmin = max(2, int(round(OCC_MIN_S * meta["fps"])))
    parts = [(0, n)] if not cut or not 0 < cut < n else [(0, cut), (cut, n)]
    counts = {}
    for k in range(n):
        q = load_pass(rd, k + 1, meta["frames"][k].get("border"), shadow=False) if occ[k] is not None else None
        if q is None:
            continue
        rgba, _, idx = q
        a = rgba[..., 3]
        fg = (a > 0.004) | (idx > 0)
        if not (fg & (veh[2] >= 0)).any():
            continue
        own = owners(idx, a, occ[k])
        h = occluded(idx, a, veh, occ[k], own)
        (y0, y1, x0, x1), code = own
        hh = h[y0:y1, x0:x1]
        if hh.any():
            u, c = np.unique(code[hh].astype(np.int64) * 256 + veh[0][y0:y1, x0:x1][hh], return_counts=True)
            for ui, ci in zip(u, c):
                counts.setdefault(int(ui), np.zeros(n, int))[k] = ci
    blocked = 0
    for ui, h in counts.items():
        for a, b in parts:
            for k in a + np.flatnonzero(short_hides(h[a:b], nmin)):
                occ[k].setdefault("block", set()).add((ui // 256, ui % 256)); blocked += 1
    return occ, blocked


def face_blur(img, heads, hidden=None):
    """Privacy mosaic on each face, as in the dataset. heads: [u, v, r25] = head-bone pixel position and the pixel
    length of 0.25 m there (build_scene meta). Face centre 0.09 m above the bone, radius 0.12 m, ~3 cells across."""
    for u, v, r25 in heads:
        ppm = r25 / 0.25
        r = max(2, int(round(0.12 * ppm))); cu, cv = int(round(u)), int(round(v - 0.09 * ppm))
        if hidden is not None and 0 <= cv < hidden.shape[0] and 0 <= cu < hidden.shape[1] and hidden[cv, cu]:
            continue  # head behind a parked vehicle
        y0, y1, x0, x1 = max(0, cv - r), min(img.shape[0], cv + r), max(0, cu - r), min(img.shape[1], cu + r)
        if y1 - y0 < 2 or x1 - x0 < 2:
            continue
        cells = 3
        roi = img[y0:y1, x0:x1]
        small = cv2.resize(roi, (cells, cells), interpolation=cv2.INTER_AREA)
        img[y0:y1, x0:x1] = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    return img


def compose(plate, p, look, rng, box, veh=None, occ=None, vstill=None, heads=(), blur_faces=False, solid=None):
    """One frame: plate (RGB uint8) + passes p (load_pass; None: nothing rendered) -> RGB uint8 and the visible index
    pass (None without passes). Outside box (content_px: black bars) the output is the plate itself and nothing is
    visible; veh (still_vehicles) / occ (the frame's occ_rows) / vstill: still parked vehicles, which hide what is
    behind them (occluded); solid: plate structure pixels (structures.Struct.solid), which get no synthetic shadow.
    The actor and distractors take look['gain'], the object and distractor objects look['obj_gain'] (D44; default the
    same)."""
    H, W = plate.shape[:2]
    inside = np.zeros((H, W), bool); inside[box[1]:box[3], box[0]:box[2]] = True
    bars = not inside.all()
    bg = srgb2lin(plate.astype(np.float32) / 255)
    hidden = vis = None
    if p is not None:
        rgba, sh, idx = p
        og = look.get("obj_gain", look["gain"])
        fga = (rgba[..., 3] > 0.004) | (idx > 0)
        own = owners(idx, rgba[..., 3], occ) if fga.any() and (og != look["gain"] or (
            veh is not None and occ is not None and (fga & (veh[2] >= 0)).any())) else None
        hidden = occluded(idx, rgba[..., 3], veh, occ, own)
        out = hidden | ~inside  # behind a parked vehicle or outside the camera image
        rgba[out] = 0.0
        if bars:
            sh = np.where(inside, sh, 1.0)  # no synthetic shadow in a bar
        if vstill is not None:
            sh = np.where(vstill, 1.0, sh)  # the shadow catcher is the ground: no shadows on vehicles
        if solid is not None:
            sh = np.where(solid, 1.0, sh)  # nor on walls, pillars, piles or the front layer (D34)
        bg = bg * sh[..., None]
        gain = np.full((H, W, 1), look["gain"], np.float32)
        if own is not None and og != look["gain"]:  # object and distractor objects (fringes with them, owners)
            (y0, y1, x0, x1), code = own
            gain[y0:y1, x0:x1, 0][(code == 2) | (code >= 20)] = og
        fg = srgb2lin(rgba[..., :3]) * gain
        if look["ir"]:
            fg = np.repeat(lum(fg)[..., None], 3, 2)
        a = rgba[..., 3:4]
        if look["blur"] > 0:
            fa = cv2.GaussianBlur(np.concatenate([fg * a, a], 2), (0, 0), look["blur"])
            a = fa[..., 3:4]; fg = fa[..., :3] / np.maximum(a, 1e-6)
        out_lin = fg * a + bg * (1 - a)
        img = lin2srgb(out_lin) * 255
        nz = rng.normal(0, look["noise"], img.shape[:2])[..., None]
        img = img + nz * (a > 0.01)
        img = np.clip(np.rint(img), 0, 255).astype(np.uint8)  # rounded (D39): truncation lowered 63 of the 256 plate
        # levels by 1, and 255 of them inside the render region (shadow pass <= 0.99986 there): a 1-level rectangle
        vis = np.where(out, 0, idx)  # visible pixels (QC)
    else:
        img = np.rint(lin2srgb(bg) * 255).astype(np.uint8)
    if blur_faces:  # a face cut by the content edge is still blurred (as a real one at the frame edge)
        img = face_blur(img, heads, hidden)
    return (np.where(inside[..., None], img, plate) if bars else img), vis


def apart_px(vis, idx, act_box):
    """Visible object pixels (vis == 2) outside the actor's pixels (index pass == 1) dilated by a disk of
    qc.apart_radius(actor box height) px."""
    obj = vis == 2
    if not obj.any() or act_box[0] < 0:
        return int(obj.sum())
    r = qc.apart_radius(act_box[3] - act_box[1] + 1)
    near = cv2.dilate((idx == 1).astype(np.uint8), qc.disk(r)).astype(bool)
    return int((obj & ~near).sum())


def main(sc_path, rd, out):
    S = json.loads(pathlib.Path(sc_path).read_text()); gr, m = S["group"], S["member"]
    rd = pathlib.Path(rd)
    meta = json.loads((rd / "meta.json").read_text())
    fps, n, (W, H) = gr["render_fps"], meta["n"], meta["size"]
    times = [k / fps for k in range(n)]
    plates = plate_frames(plate_path(gr["plate_id"]), gr["plate_start"], times, (W, H))
    box = content_px(group_calib(gr), (W, H))  # the camera the scenario was sampled with
    vf = SYN / f"data/vehicles/{gr['plate_id']}.png"  # parked vehicles occlude what is behind them
    vlab = cv2.imread(str(vf), cv2.IMREAD_UNCHANGED) if vf.exists() else None
    vj = SYN / "data/vehicles.json"
    vrec = [v for v in (json.loads(vj.read_text()).get(gr["plate_id"], []) if vj.exists() else [])
            if v.get("n_frames", 0) == v.get("n_of", -1)]  # still for the whole window
    vbottom = {v["id"]: v["bottom_v"] for v in vrec}
    vstill = np.isin(vlab, list(vbottom)) if vlab is not None else None
    solid = None
    if gr.get("structures"):  # sampled with structure masks (D34): the masks it was planned with
        st = structures.load(gr["plate_id"])
        if st is None:
            raise RuntimeError(f"{gr['plate_id']}: no structure raster (plate_structures.py raster)")
        if st.meta["sha"] != gr["structures"].get("sha"):
            print(f"WARNING {gr['group_id']}: structure masks changed since sampling "
                  f"({gr['structures'].get('sha')} -> {st.meta['sha']})", flush=True)
        if st.lab.shape != (H, W):
            raise RuntimeError(f"{gr['plate_id']}: structure raster {st.lab.shape[::-1]} != render size {(W, H)}")
        solid = st.solid
    lf = rd.parent / "look.json"
    look = json.loads(lf.read_text()) if lf.exists() else None
    if look is None or (look.get("v") != LOOK_V and m.get("y", 1) == 1):  # the positive remakes an older look (D46)
        look = dict(make_look(gr, m, rd, meta, plates, box, vlab, solid), v=LOOK_V); lf.write_text(json.dumps(look))
    rng = np.random.default_rng(gr["seed"] + 11)
    vcols = {}
    for kv in vbottom:  # per column lowest mask row of each still vehicle
        m_ = vlab == kv
        vcols[kv] = np.where(m_, np.arange(m_.shape[0])[:, None], -1).max(0)
    cam = Camera(group_calib(gr), (W, H))
    veh = occ = None; nblock = 0
    if vcols:  # 3D solids fitted to the masks; the walkers' ground positions per frame (D44, D45)
        solids = plate_solids(gr["plate_id"], group_calib(gr), (W, H), vlab,
                              [(v["id"], v["cls"], v["contact_uv"]) for v in vrec])
        veh = still_vehicles(vlab, vcols, vehicle_depth(cam, vlab, solids))
        dv = [d for d in [m.get("divergence")] + [json.loads(f.read_text())["member"].get("divergence") for f in
                                                   pathlib.Path(sc_path).parent.glob(f"{gr['group_id']}_*.json")] if d]
        cut = int(np.floor(min(dv) * fps)) if dv else None  # the group's divergence (the negative carries it)
        occ, nblock = occ_block(occ_rows(gr, m, cam, times), veh, rd, meta, cut)
    occ = occ or [None] * n
    obj_px, act_px, obj_apart = np.zeros(n, int), np.zeros(n, int), np.zeros(n, int)
    act_box = np.full((n, 4), -1, int)  # actor (index 1) pixel box x0, y0, x1, y1 per frame; -1 if not visible
    obj_box = np.full((n, 4), -1, int)  # visible object pixel box
    with av.open(str(out), "w") as c:
        st = c.add_stream("libx264", rate=Fraction(fps).limit_denominator(1001))
        st.width, st.height, st.pix_fmt = W, H, "yuv420p"
        st.options = {"crf": "14", "preset": "medium"}
        for k in range(n):
            p = load_pass(rd, k + 1, meta["frames"][k].get("border"))
            img, vis = compose(plates[k], p, look, rng, box, veh, occ[k], vstill, meta["frames"][k].get("heads", []),
                               gr["face_blur"], solid)
            if vis is not None:
                obj_px[k], act_px[k] = int((vis == 2).sum()), int((vis == 1).sum())
                idx = p[2]
                if (idx == 1).any():  # the box keeps the whole body (height QC)
                    ys, xs = np.nonzero(idx == 1); act_box[k] = [xs.min(), ys.min(), xs.max(), ys.max()]
                if obj_px[k]:
                    ys, xs = np.nonzero(vis == 2); obj_box[k] = [xs.min(), ys.min(), xs.max(), ys.max()]
                    obj_apart[k] = apart_px(vis, idx, act_box[k])
            fr = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24")
            fr.pts = k
            for pkt in st.encode(fr):
                c.mux(pkt)
        for pkt in st.encode():
            c.mux(pkt)
    np.savez_compressed(rd / "masks.npz", obj_px=obj_px, act_px=act_px, act_box=act_box, obj_box=obj_box,
                        obj_apart_px=obj_apart, obj_at_rest=np.array([f["obj_at_rest"] for f in meta["frames"]]),
                        occ_blocked=nblock)
    return look


if __name__ == "__main__":
    print(main(*sys.argv[1:4]))
