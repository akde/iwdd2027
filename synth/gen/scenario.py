"""Minimal-pair scenario sampling (pure Python; no Blender).
python synth/gen/scenario.py --groups N --seed S [--families place_leave] -> synth/data/scenarios/<group>_<outcome>.json

A group is one shared scenario: plate + window, camera (snapshot of the plate's calib, gr['cam']), actor (avatar,
height, walk), object, path (entry E -> spot X -> exit B), timing (t_event = start of the disposal motion), duration,
light, distractors, face blur, degradation seed. Its members differ only in the outcome (invariant 1); each group
renders one positive and one negative.
Outcomes:
  leave       (place_leave, y=1): carry in, bend, release at the bottom, stand, walk out; the object stays.
  pickup      (place_pickup, y=0): as leave up to standing; pause; bend again, grasp, stand, walk out carrying.
  carry       (carry_through, y=0): as leave up to t_event; stands instead of bending, walks out carrying.
  preexisting (pickup_preexisting, y=0): object at the drop point from frame 0; actor arrives empty-handed on the
              same path and timing, bends at t_event, grasps at the leave member's release time, walks out carrying.
Members are identical before the divergence time (stand-up end for pickup, t_event for carry; preexisting differs
from frame 0 by the object on the ground).
Labels: t_onset_exact = t_event + DELTA; timestamp = quantize(t_onset_exact) for y = 1, else -1 (DECISIONS.md).
t_dec = first time the outcome is decidable: actor >= 2 m from the object for 1 s (leave, carry), or object back in
hand (pickup, preexisting).
Walkers and the resting object (F1, DECISIONS.md D33): the object rests at a point fixed in the actor's frame
(rest_box); plan_path draws only exits whose leg corridor stays clear of it (exit_clear; a narrow track may turn back).
run_batch then measures every member's legs in Blender before rendering (qc.leg_verdict).
Distractors (F3, DECISIONS.md D36; plan_distractors): drawn after the members exist, only on approved lines of the
distractor line library (synth/data/distractor_lines.json, synth/scripts/distractor_lines.py; default deny, >=
MIN_LINES per plate), keeping DIS_OBJ from the spot and the rest box, WALK_CLEAR from every member's actor and from each
other while in view, and passing the strict screen on their own track; a distractor that cannot is left out.
Plate structures (F4, DECISIONS.md D34; gen/structures.py): only vetted plates with structure masks are sampled
(structures.usable, fail-closed); the spot X (feet and body), the object's rest box, the entry and exit points, the
in-view path with its walk-in / walk-out extensions (clear), the car-mode place S, S -> X and the exits, and every
distractor's track stay off the masks (feet not on F / N or behind S, body not over F or behind S); gr['structures']
records the hash of the masks the group was planned with (the compositor reads it).
D35: female avatars bend with the male crouch (her own set the object down at her right foot, RC2; rest_box keeps the
old point for older scenarios); car-mode clips last until both members have turned and walked off (walkout_after_event);
Tables counts accepted groups per scene and mode for run_batch's scene cap (scenes) and marks night plates
(plate_night) for its per-group night flag.
D38 (review fixes): the planners also keep the carried object (structures.carry_box, every member's, on the walked
legs and the turns) and the crouch reach at X (structures.crouch_conflict) off the structure masks; car-mode groups get
distractors as on-foot groups do; car-mode clips last until the positive is back at the car + 1 s in every negative
family (vehicle_need_after_event); a sampler that finds no plan raises NoPlan with the families and gender it drew.
D40 (light): the sun follows the plate. synth/data/plate_light.csv (synth/scripts/plate_light.py: real cast shadows
annotated on reference sheets, converted to world azimuth / elevation with the plate's camera) gives each plate a
condition: sun (the annotated direction, jittered per group), overcast or night (no sun). Plates without a usable row
are not sampled unless Tables(light_fallback=True) (run_batch --light-fallback), which keeps the old random light."""
import argparse, collections, json, math, pathlib, sys
import numpy as np, pandas as pd

from camera import Camera, inside, plate_calib, snapshot
from tracks import track, walker_keys
import qc, structures

SYN = pathlib.Path(__file__).resolve().parents[1]
ROOT = SYN.parent
STAND_Z = {"m": 0.895, "f": 0.919}
WALKS = {"m": ["m_walk_neutral", "m_walk_neutral_01", "m_walk_neutral_02", "m_walk_slow_01", "m_walk_slow_02",
               "m_walk_stroll_01", "m_walk_cool_01", "m_walk_fast_01"],
         "f": ["f_walk_neutral", "f_walk_neutral_01", "f_walk_neutral_02", "f_walk_neutral_03", "f_walk_slow_01",
               "f_walk_slow_02", "f_walk_stroll_01", "f_walk_self-assured"]}
IDLE = {"m": "m_idle_neutral_01", "f": "f_idle_neutral_01"}
# female avatars bend with the male crouch (RC2, DECISIONS.md D35): the female clip sets the object down at her own
# right foot (standing contact in about half of the female draws, every large bag); retargeting is armature-space
# (D12) and motion.pose_table scales a clip by the root height of the skeleton it was recorded on
CROUCH_IN = {"m": "m_crouch_in", "f": "m_crouch_in"}
CROUCH_OUT = {"m": "m_crouch_out", "f": "m_crouch_out"}
HEIGHT = {"m": (1.76, 0.07), "f": (1.63, 0.06)}
CARRY_OBJECTS = ["trashbag", "compost_bag_02", "cement_bag", "cardboard_box_01", "plastic_crate_01",
                 "plastic_crate_03", "plastic_jerrycan", "metal_jerrycan", "old_tyre", "industrial_pastic_container",
                 "wicker_basket_01"]
# (asset, scale, colours, weight): bags dominate real dumping (taxonomy: plastic carrier bag 0.45, garbage bag 0.20)
OBJECT_LAW = [("trashbag", 0.55, "light", 0.30),  # carrier-bag stand-in: small, white/light plastic
              ("trashbag", 1.0, "dark", 0.25), ("compost_bag_02", 1.0, "dark", 0.07), ("cement_bag", 1.0, None, 0.05),
              ("cardboard_box_01", 1.0, None, 0.08), ("plastic_crate_01", 1.0, None, 0.03),
              ("plastic_crate_03", 1.0, None, 0.03), ("industrial_pastic_container", 1.0, None, 0.03),
              ("plastic_jerrycan", 1.0, None, 0.04), ("metal_jerrycan", 1.0, None, 0.03), ("old_tyre", 1.0, None, 0.05),
              ("wicker_basket_01", 1.0, None, 0.04)]
LIGHT = [[0.85, 0.85, 0.82], [0.75, 0.78, 0.8], [0.8, 0.55, 0.3], [0.7, 0.75, 0.85], [0.85, 0.8, 0.65]]
DARK = [[0.02, 0.02, 0.02], [0.03, 0.05, 0.12], [0.05, 0.12, 0.05], [0.35, 0.35, 0.35], [0.1, 0.1, 0.12]]
NEG_WEIGHTS = {"place_pickup": 0.3, "carry_through": 0.5, "pickup_preexisting": 0.2,  # D19
               "unload_reload": 0.3, "carry_from_car": 0.4, "load_preexisting": 0.3}  # D26
FACE_BLUR = 0.47  # taxonomy: faces blurred in 77 of 161 clips with a visible face
BAG_COLOURS = [[0.02, 0.02, 0.02], [0.03, 0.05, 0.12], [0.85, 0.85, 0.82], [0.05, 0.12, 0.05], [0.35, 0.35, 0.35]]
NEG_OF = {"place_leave": ["place_pickup", "carry_through", "pickup_preexisting"],
          "unload_vehicle_leave": ["unload_reload", "carry_from_car", "load_preexisting"]}
OUTCOME = {"place_leave": "leave", "place_pickup": "pickup", "carry_through": "carry", "pickup_preexisting": "preexisting",
           "unload_vehicle_leave": "leave", "unload_reload": "pickup", "carry_from_car": "carry",
           "load_preexisting": "preexisting"}
# median(g - onset_seen_s) over high-confidence positives (synth/reports/taxonomy_stats.md, DECISIONS.md D20):
# per family when n >= 10, else per event type
DELTA_FAMILY = {"place_leave": 0.76, "unload_vehicle_leave": 1.00, "throw_from_car": 0.68}
DELTA_EVENT = {"static": 1.28, "dynamic": 0.53}
EVENT = {"place_leave": "static", "unload_vehicle_leave": "static", "bulky_carry_leave": "static",
         "drop_walk": "dynamic", "throw_walk": "dynamic", "throw_from_car": "dynamic"}
QUANT = "round"
DELTA = DELTA_FAMILY["place_leave"]


def delta(family):
    return DELTA_FAMILY.get(family, DELTA_EVENT[EVENT[family]])
MAX_USES = 20
# swimwear and football kits look absurd at a dumping site (avatar lineup inspection, DECISIONS.md D18)
EXCLUDE_AVATARS = {"Sports_Female_01", "Sports_Male_01", "Sports_Male_02", "Sports_Male_03"}
MIN_PX = 0.06  # the dumping spot: a 1.7 m person is >= 6% of the image height (event visible after degradation)
POS_UP = 0.3  # every member has finished its last stand-up >= POS_UP s before the clip ends (F5, DECISIONS.md D32)
# F1 (D33): the released object rests under the right palm at the release (+ 0.06 m away from the pelvis,
# build_scene.bake_object); with one crouch per gender that point is fixed in the actor's frame: (forward, left) m at
# scale 1, Blender-measured over all avatars of the gender (one skeleton per gender, std < 1 mm; female with the male
# crouch over her 22 avatars, D35); the gate re-measures it on every member (run_batch)
REST_OFF = {"m": (0.531, -0.126), "f": (0.491, -0.082)}
REST_OFF_CROUCH = {("f", "f_crouch_in"): (0.253, -0.228)}  # scenarios before D35 (female crouch): at her right foot
# set-down distance (D41, step D3): the crouch gate (qc.crouch_th) found the knees and shins inside the large objects
# in most draws (cardboard box 76%, container 86%, female cement bag 86%); these are put down SETDOWN m further forward
# (the actor's facing) than under the palm (build_scene.bake_object: the offset grows as the hand goes down, so the
# bend carries the object forward); each as far as the palm stays over the object's top face at the release
# (0.06 + d <= half its depth at scale 0.9); rest_box adds it; every other object: 0 (as before D41)
SETDOWN = {"cardboard_box_01": 0.10, "industrial_pastic_container": 0.14, "cement_bag": 0.14, "old_tyre": 0.10}
# sun per plate from real cast shadows (D40, synth/scripts/plate_light.py -> plate_light.csv): sun iff the plate's
# annotated condition is 'sun'; per group the annotated azimuth + U(-AZ_JIT, AZ_JIT) and elevation + U(-EL_JIT, EL_JIT)
# (EL_LAW when no shadow of known height was annotated); a plate without a usable row is not sampled (fail-closed)
# unless Tables(light_fallback=True), which keeps the old law for it (SUN_P on day plates, azimuth U(0, 360), EL_LAW)
LIGHT_CSV = SYN / "data/plate_light.csv"
LIGHT_CONDITIONS = ("sun", "overcast", "night")
SUN_P, EL_LAW = 0.6, (25.0, 65.0)
AZ_JIT, EL_JIT = 8.0, 5.0
NIGHT_SHARE = 0.10  # real contest clips at night; run_batch: night probability per group in a mode with night plates
NIGHT_MIN_SCENES = 2  # a mode draws night groups only if its usable night plates span >= 2 scenes (car mode: 1)
SCENE_CAP = 2.0  # run_batch: a scene takes a group only while it stays <= 2x the mean over the mode's scenes in use
LEG_A0, LEG_W, LEG_MARGIN = 0.25, 0.15, 0.03  # walk-out leg corridor: from 0.25 m ahead of X, +-0.15 m wide, 3 cm clear
DIS_CLEAR = 1.0  # m from a distractor's in-clip line to X and to the rest centre (D33; plan_distractors: DIS_OBJ)
WALK_CLEAR, WALK_CLEAR_CARRY = 0.6, 1.2  # m root to root between walkers in view (1.2 m when either carries an object)
# F3 (D36): distractors walk only approved lines of the line library (synth/scripts/distractor_lines.py: straight
# crossings checked against the walkable polygon, the camera image, the structure masks, parked and moving vehicles;
# every line 'pending' until it is approved on its overlay sheet: default deny)
LINES = SYN / "data/distractor_lines.json"
MIN_LINES = 3  # approved lines a plate needs before it gets distractors (one corridor would repeat on every group)
DIS_LAW = (0.55, 0.30, 0.15)  # P(0, 1, 2 distractors), drawn per group
DIS_TRIES = 30  # tries per distractor (line, direction, timing, walker); a distractor that fails them all is left out
DIS_OBJ = 1.2  # m from a distractor's in-clip line to X and to the object's rest box (stress test in D36)
DIS_VIS = 2.0  # s: a distractor's feet are in view for >= min(DIS_VIS, its crossing) inside the clip
DIS_H = (1.45, 1.95)  # m: distractor heights (normal law of the gender, clipped)
DIS_SEED = 0xD15  # plan_distractors' random stream: (group seed, DIS_SEED), apart from the sampler's
_LIB = {}
# s: the walk-in / walk-out keys reach 60 s beyond E and B (plan_member, walker_keys); the planners test the in-view
# part that can fall inside the clip: min(STRUCT_EXT, plate length) s at the walker's speed
STRUCT_EXT = 60.0
STRUCT_STEP = 0.1  # m between tested points (the screen tests the tracks every 0.1 s, about 0.1-0.15 m)
DMIN_SMALL = 2.5  # m: entry/exit distance when no 4 m plan exists on a masked plate (small views; car mode walks 2.5-7 m)
_OBJECTS = None


def quantize(t, rule=QUANT):
    return int(math.floor(t)) if rule == "floor" else int(math.floor(t + 0.5))


class NoPlan(RuntimeError):
    """A sampler found no plate or path. It carries what the draw had already drawn (the [positive, negative] families
    and the actor's gender), so run_batch.plan keeps them over the group's redraws as after a screen rejection (D38,
    review R2: a failed first draw let the redraw pick a new family and gender)."""

    def __init__(self, msg, families, gender):
        super().__init__(msg)
        self.families, self.gender = [str(f) for f in families], str(gender)


class Tables:
    def __init__(self, light_fallback=True, light_csv=LIGHT_CSV):
        """light_fallback: plates without a usable plate_light.csv row keep the old random light (library callers and
        tests); run_batch and this module's CLI pass False unless --light-fallback (fail-closed, D40)."""
        self.clip_meta = pd.read_csv(SYN / "data/taxonomy/clips.csv").set_index("video")  # contest label metadata
        self.plates = pd.read_csv(SYN / "data/plates.csv", dtype={"label": str})
        chk = pd.read_csv(SYN / "data/calib_check.csv")  # stage-2 render check (synth/scripts/check_calib.py)
        ok = set(chk[chk.ok].plate_id)
        self.plates = self.plates[[p in ok and (SYN / f"calib/{p}.json").exists() for p in self.plates.plate_id]]
        # structure masks and plate vetting, fail-closed (F4 / F6 part B, D34): plate id -> why it is not sampled
        why = structures.usable(self.plates, lambda p: plate_calib(p)["render_size"])
        self.excluded = dict(zip(self.plates.plate_id[why != ""], why[why != ""]))
        self.plates = self.plates[why == ""].reset_index(drop=True)
        self.plates["night"] = [plate_night(self.clip_meta, pr) for pr in self.plates.itertuples()]
        # light per plate (D40): unlit = plate id -> why its row is missing or unusable; fail-closed without fallback
        self.light, self.light_fallback = load_light(light_csv), light_fallback
        why = np.array([self.light_why(pr) for pr in self.plates.itertuples()], dtype=object)
        self.unlit = dict(zip(self.plates.plate_id[why != ""], why[why != ""]))
        if not light_fallback:
            self.excluded.update(self.unlit)
            self.plates = self.plates[why == ""].reset_index(drop=True)
        self.anim = pd.read_csv(SYN / "data/anim_inventory.csv").set_index("name")
        self.avatars = pd.read_csv(SYN / "data/avatar_table.csv")
        self.avatars = self.avatars[~self.avatars.avatar.isin(EXCLUDE_AVATARS)].reset_index(drop=True)
        self.objects = pd.read_csv(SYN / "data/object_table.csv").set_index("asset")
        idx = pd.read_csv(ROOT / "cache/index.csv")
        self.real_T = idx["T"].to_numpy()
        self.real_g = idx[idx.y == 1].g.to_numpy(float)
        self.uses = {p: 0 for p in self.plates.plate_id}
        # per-batch scene counts by mode ('foot' / 'car'), accepted draws only; scene_cap (run_batch) caps them
        self.scene_uses, self.scene_cap, self.cap_lifted = collections.Counter(), None, 0
        # real people's size: per-clip median YOLO11s box height / frame height in positive contest clips
        rel = []
        for v in idx[idx.y == 1].video:
            b = np.load(SYN / f"data/det/{v}.npz")["boxes"]
            b = b[(b[:, 5] >= 0.5) & (b[:, 2] > 0.01) & (b[:, 4] < 0.99)]
            if len(b):
                rel.append(float(np.median(b[:, 4] - b[:, 2])))
        self.real_rel_h = np.array(rel)
        self.obstacles = json.loads((SYN / "data/obstacles.json").read_text())  # synth/scripts/plate_obstacles.py
        self.vehicles = json.loads((SYN / "data/vehicles.json").read_text())  # synth/scripts/plate_vehicles.py
        self._vlab, self._struct, self._lines = {}, {}, {}
        self.lib = load_lines()  # distractor line library (F3, D36); {} without the file: no distractors

    def light_why(self, pr):
        """Why plate row pr has no usable light row ('' = it has one): none, a camera changed since the annotation,
        or a row that contradicts the plate's night / dim labels (light_conflict)."""
        r = self.light.get(pr.plate_id)
        if r is None:
            return "no light annotation"
        if r.get("cam") != cam_sha(self.calib(pr.plate_id)):
            return "light annotation stale (camera changed)"
        return light_conflict(r.get("condition"), r.get("sun_az_deg"), pr.night, plate_dim(self.clip_meta, pr))

    def light_row(self, plate_id):
        """The plate's plate_light.csv row, or None (only with light_fallback: the old random light)."""
        return None if plate_id in self.unlit else self.light[plate_id]

    def lines(self, plate_id):
        """The plate's approved distractor lines (approved_lines; [] = no distractors on it)."""
        if plate_id not in self._lines:
            self._lines[plate_id] = approved_lines(self.lib, plate_id, self.calib(plate_id), self.struct(plate_id))
        return self._lines[plate_id]

    def still_car(self, plate_id):
        """The plate has a vehicle parked for its whole window (car mode)."""
        return any(v.get("n_frames", 0) == v.get("n_of", -1) for v in self.vehicles.get(plate_id, []))

    def take(self, pr, mode):
        """Count an accepted draw on plate row pr (MAX_USES per plate, the scene cap per mode)."""
        self.uses[pr.plate_id] += 1; self.scene_uses[(mode, pr.scene_id)] += 1

    def unuse(self, gr):
        """Undo take() for a drawn group the screen rejected (run_batch)."""
        self.uses[gr["plate_id"]] -= 1; self.scene_uses[(mode_of(gr), gr["scene_id"])] -= 1

    def scenes(self, plates, mode):
        """Scenes of `plates` a new group of `mode` may be drawn from (uniform choice). With scene_cap (run_batch
        --scene-cap), a scene is left out while one more group would put it above scene_cap x the mean count over the
        scenes this mode has used so far in the batch (the uniform share over the scenes that can plan: car mode has
        16 scenes with parked cars, but plans on about 9); if that leaves none, every scene (counted in cap_lifted)."""
        sc = sorted(plates.scene_id.unique())
        used = [c for (m, _), c in self.scene_uses.items() if m == mode and c > 0]
        if not self.scene_cap or not used:
            return sc
        cap = self.scene_cap * (sum(used) + 1) / len(used)
        ok = [x for x in sc if self.scene_uses[(mode, x)] + 1 <= cap]
        if not ok:
            self.cap_lifted += 1
        return ok or sc

    def night_feasible(self, car, min_scenes=NIGHT_MIN_SCENES):
        """Night groups can be drawn in this mode: its usable night plates span >= min_scenes scenes."""
        p = self.plates[self.plates.night.astype(bool)]
        if car:  # (astype: an empty mapped mask is object dtype and would select columns; a pilot list without night)
            p = p[p.plate_id.map(self.still_car).astype(bool)]
        return p.scene_id.nunique() >= min_scenes

    def vlab(self, plate_id):
        if plate_id not in self._vlab:
            import cv2
            self._vlab[plate_id] = cv2.imread(str(SYN / f"data/vehicles/{plate_id}.png"), cv2.IMREAD_UNCHANGED)
        return self._vlab[plate_id]

    def calib(self, plate_id):
        return plate_calib(plate_id)

    def struct(self, plate_id):
        """The plate's structure masks (structures.Struct); every sampled plate has them (usable)."""
        if plate_id not in self._struct:
            st = structures.load(plate_id)
            if st is None:
                raise RuntimeError(f"no structure raster for {plate_id} (plate_structures.py raster)")
            self._struct[plate_id] = st
        return self._struct[plate_id]


def plate_night(clip_meta, pr):
    """Night plate: the contest clip's label (time_of_day); MEVA by clock time (the 23:55 clips)."""
    if pr.source == "contest":
        return bool(clip_meta.loc[pr.video].time_of_day == "night")
    return "_23-55-" in pr.video


def plate_dim(clip_meta, pr):
    """Dim plate: the contest clip's label (lighting low); MEVA by the plate's median luma (< 50)."""
    return bool(clip_meta.loc[pr.video].lighting == "low") if pr.source == "contest" else bool(pr.luma < 50)


def cam_sha(cal):
    """Hash of the camera fields a light annotation was converted with (plate_light.py convert)."""
    import hashlib
    return hashlib.sha1(json.dumps([cal["camera_world"], cal["focal_px"], cal["image_size"]]).encode()).hexdigest()[:12]


def load_light(path=LIGHT_CSV):
    """plate_light.csv (synth/scripts/plate_light.py convert): plate id -> its row (str fields; '' = empty)."""
    path = pathlib.Path(path)
    if not path.exists():
        return {}
    rows = pd.read_csv(path, dtype=str, keep_default_na=False).to_dict("records")
    return {p: r for r in rows for p in r["plates"].split()}


def light_conflict(condition, az, night, dim):
    """Why a light condition does not fit a plate ('' = it does): 'night' exactly on night plates, no sun on dim plates
    (the night / dim / lamp logic stays as it was), and a sun row needs its azimuth. plate_light.py convert and Tables
    both apply it."""
    if condition not in LIGHT_CONDITIONS:
        return f"bad light condition {condition!r}"
    if (condition == "night") != bool(night):
        return f"light condition {condition} on a {'night' if night else 'day'} plate"
    if condition == "sun" and dim:
        return "sun on a dim plate"
    if condition == "sun" and az in (None, ""):
        return "sun without an azimuth"
    return ""


def group_light(rng, t, pr, night, dim):
    """The group's light (both members share it): from the plate's plate_light.csv row (D40), or the old law when
    the plate has none (Tables light_fallback only). Every branch draws the same three numbers before the lamp, so the
    rest of the group (plate start, seeds, face blur) does not depend on the table."""
    r = t.light_row(pr.plate_id)
    coin = rng.random()
    if r is not None and r["condition"] == "sun":
        sun, az = True, (float(r["sun_az_deg"]) + rng.uniform(-AZ_JIT, AZ_JIT)) % 360.0
        el = (float(np.clip(float(r["sun_el_deg"]) + rng.uniform(-EL_JIT, EL_JIT), 5.0, 85.0)) if r["sun_el_deg"]
              else rng.uniform(*EL_LAW))
    else:  # overcast / night rows; without a row the old law (sun on day plates with SUN_P)
        sun, az, el = r is None and bool(coin < (0.0 if night or dim else SUN_P)), rng.uniform(0, 360), rng.uniform(*EL_LAW)
    return dict(sun=bool(sun), sun_az=float(az), sun_el=float(el), night=night, dim=dim, ir=bool(pr.chroma < 2.0),
                lamp=bool(night and rng.random() < 0.7), source="fallback" if r is None else "table",
                condition=None if r is None else r["condition"])


def mode_of(gr):
    return "car" if "vehicle" in gr else "foot"


def load_lines(path=LINES):
    """The distractor line library (synth/scripts/distractor_lines.py), cached per path; {} if there is none."""
    path = pathlib.Path(path)
    if path not in _LIB:
        _LIB[path] = json.loads(path.read_text()) if path.exists() else {}
    return _LIB[path]


def calib_hash(cal):
    """Hash of the camera fields a library line was built with (camera, sizes, walkable polygon, content box)."""
    import hashlib
    k = {f: cal.get(f) for f in ("camera_world", "focal_px", "image_size", "render_size", "walkable", "content")}
    return hashlib.sha1(json.dumps(k, sort_keys=True).encode()).hexdigest()[:12]


def approved_lines(lib, plate_id, cal, st):
    """A plate's distractor lines with status 'approved' (default deny: 'pending' and 'rejected' lines are never
    used), only while the library entry was built with the plate's current camera (calib_hash) and structure masks
    (st.meta sha); otherwise none (fail-closed: rebuild the library, which keeps the statuses of unchanged lines)."""
    e = (lib or {}).get("plates", {}).get(plate_id)
    if not e or e.get("calib") != calib_hash(cal) or st is None or e.get("structures") != st.meta["sha"]:
        return []
    return [ln for ln in e["lines"] if ln.get("status") == "approved"]


def walk_speed(t, anim, avatar, scale):
    a = t.avatars.set_index("avatar").loc[avatar]
    return float(t.anim.loc[anim, "speed_mps"] * a.root_z / STAND_Z[a.gender] * scale)


def boundary_points(poly, step=0.5):
    poly = np.asarray(poly, float)
    out = []
    for a, b in zip(poly, np.roll(poly, -1, 0)):
        n = max(1, int(np.linalg.norm(b - a) / step))
        out += [a + (b - a) * k / n for k in range(n)]
    return np.array(out)


def path_points(E, X, B, ext=10.0, step=0.25):
    """Ground points every `step` m along E -> X -> B, continued `ext` m beyond E and B (walk-in / walk-out)."""
    out = []
    for a, b, pre, post in ((E, X, ext, 0.0), (X, B, 0.0, ext)):
        L = np.linalg.norm(b - a); u = (b - a) / max(L, 1e-9)
        out.append(a + np.arange(-pre, L + post + 1e-9, step)[:, None] * u)
    return np.vstack(out)


def leg_points(A, B, pre=0.0, post=0.0, step=STRUCT_STEP):
    """Ground points every `step` m along A -> B, continued `pre` m before A and `post` m beyond B."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    L = np.linalg.norm(B - A); u = (B - A) / max(L, 1e-9)
    return A + np.arange(-pre, L + post + 1e-9, step)[:, None] * u


def carry_clear(cam, st, legs, turns, carry):
    """No carried object (structures.carry_conflict, planner padding) meets a structure on the walked legs [(A, B, pre,
    post)] (facing A -> B, the walking envelope) or the in-place turns [(point, from, to, walking)]
    (structures.turn_arc; walking: the turn starts on arrival, within CARRY_SETTLE of the walk); carry =
    dict(scale, obj) (D38, review G1/S4)."""
    for A, B, pre, post in legs:
        P = leg_points(A, B, pre, post)
        a = math.atan2(*(np.subtract(B, A))[::-1])
        if structures.carry_conflict(cam, st, P, np.full(len(P), a), carry["scale"], carry["obj"], structures.PLAN_PAD).any():
            return False
    for P, a, b, walking in turns:
        h = structures.turn_arc(a, b)
        if structures.carry_conflict(cam, st, np.repeat(np.asarray(P, float)[None], len(h), 0), h, carry["scale"],
                                     carry["obj"], structures.PLAN_PAD, walking).any():
            return False
    return True


def clear(cam, E, X, B, obst, st=None, height=1.8, ext=STRUCT_EXT * 1.5, carry=None):
    """The path E -> X -> B keeps its feet off parked vehicles (10 m beyond E and B) and, with structure masks st,
    every point of it and of its walk-in / walk-out extensions (ext m beyond E and B) that is in view is free of
    structures for a walker of `height` m (structures.walk_conflict: feet along the line, body boxes every
    STRUCT_STEP m); with carry (dict(scale, obj), D38) also the object carried in on E -> X, out on X -> B and while
    turning at X (some member carries it on each: carry_clear)."""
    if obst is not None and qc.feet_on_vehicle(cam, path_points(E, X, B), obst["vehicles"]) > 0:
        return False
    if st is None:
        return True
    if any(structures.walk_conflict(cam, st, path_points(E, X, B, ext=ext, step=STRUCT_STEP), height, plan=True)):
        return False
    if carry is None:
        return True
    a_in, a_out = math.atan2(*(X - E)[::-1]), math.atan2(*(B - X)[::-1])
    return carry_clear(cam, st, [(E, X, ext, 0.0), (X, B, 0.0, ext)], [(X, a_in, a_out, False)], carry)


def rot(a):
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def object_height(asset, scale, objects=None):
    """Height (m) of an object's box (object_table.csv dz) at its scale."""
    object_half(asset, scale, objects)  # loads the table
    return float((objects if objects is not None else _OBJECTS).loc[asset, "dz"] * scale)


def object_half(asset, scale, objects=None):
    """Half extents (x, y) m of an object's ground box (object_table.csv dx, dy: centred meshes) at its scale."""
    global _OBJECTS
    if objects is None:
        if _OBJECTS is None:
            _OBJECTS = pd.read_csv(SYN / "data/object_table.csv").set_index("asset")
        objects = _OBJECTS
    return np.array([objects.loc[asset, "dx"], objects.loc[asset, "dy"]], float) / 2 * scale


def carried_obj(gr, objects=None):
    """The group's object as structures.carry_box carries it: (half extents [2] m, height m, yaw deg)."""
    O = gr["object"]
    return object_half(O["asset"], O.get("scale", 1.0), objects), object_height(O["asset"], O.get("scale", 1.0), objects), O["yaw"]


def distractor_carry(D):
    """(scale, carry_box obj spec) of a distractor's carried object (build_scene: scale 1, yaw 0), None without one."""
    return None if not D.get("carry") else (D["scale"], (object_half(D["carry"], 1.0), object_height(D["carry"], 1.0), 0.0))


def rest_box(X, a_in, gender, scale, half, yaw_deg, crouch=None, setdown=0.0):
    """Where the released object rests (F1, D33): (centre [2], angle rad, half extents [2]) of its ground box for an
    actor of `gender` and `scale` that arrived at X facing a_in; the object is turned by a_in + yaw (bake_object).
    Pre-existing objects lie at the same point (ground_at_release). Error <= 1 cm against Blender (6-15 fps).
    crouch: the actor's crouch_in clip (scenarios before D35 bent female avatars with f_crouch_in); setdown: the
    object's set-down distance (m forward, SETDOWN, D41)."""
    off = REST_OFF_CROUCH.get((gender, crouch), REST_OFF[gender])
    c = np.asarray(X, float) + rot(a_in) @ (np.array(off) * scale + np.array([setdown, 0.0]))
    return c, a_in + math.radians(yaw_deg), np.asarray(half, float)


def group_rest_box(gr, objects=None):
    """rest_box of a scenario group (on foot and car mode: the actor walks E -> X)."""
    E, X = np.array(gr["path"]["E"], float), np.array(gr["path"]["X"], float)
    A, O = gr["actor"], gr["object"]
    return rest_box(X, math.atan2(*(X - E)[::-1]), A["gender"], A["scale"],
                    object_half(O["asset"], O.get("scale", 1.0), objects), O["yaw"], A.get("crouch_in"),
                    setdown=O.get("setdown", 0.0))


def box_depth(P, box):
    """Largest penetration (m, > 0 inside) of ground points P [N,2] into a rest box; < 0: all at least that far out
    along one box axis."""
    c, th, h = box
    L = (np.atleast_2d(P) - c) @ rot(th)
    return float(np.minimum(h[0] - np.abs(L[:, 0]), h[1] - np.abs(L[:, 1])).max())


def exit_clear(X, B, box, a0=LEG_A0, w=LEG_W, margin=LEG_MARGIN, step=0.03):
    """The straight walk-out X -> B keeps its leg corridor (from a0 m ahead of X to beyond the box, +-w m wide)
    `margin` m outside the rest box. The first a0 m are left out: the female rest point lay at her right foot before
    D35, so no exit would pass; the gate measures contact at X."""
    X, B = np.asarray(X, float), np.asarray(B, float)
    d = np.linalg.norm(B - X)
    if d < 1e-6:
        return False
    u = (B - X) / d; n = np.array([-u[1], u[0]])
    a1 = max(a0, np.linalg.norm(box[0] - X) + np.linalg.norm(box[2]) + 0.3)
    A, W = np.meshgrid(np.arange(a0, a1 + 1e-9, step), np.arange(-w, w + 1e-9, step))
    return box_depth(X + A.ravel()[:, None] * u + W.ravel()[:, None] * n, box) < -margin


def seg_point_dist(P, A, B):
    """Distance from point P to the segment A-B."""
    P, A, B = (np.asarray(v, float) for v in (P, A, B))
    AB = B - A; s = np.clip(np.dot(P - A, AB) / max(np.dot(AB, AB), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(A + s * AB - P))


def plan_path(rng, cal, cam, tries=200, dmin=4.0, dmax=15.0, target=None, obst=None, box_of=None, st=None,
              height=1.8, speed=1.5, obj_h=0.6, span=STRUCT_EXT, carry=None):
    """Spot X inside the walkable polygon with the whole person visible and >= MIN_PX tall (with `target`, a
    relative person height: X among the candidates closest to it, so actor sizes follow real people's), the object's
    rest area out of black bars (qc.rest_in_image); entry E and exit B on the polygon boundary, dmin..dmax m from X
    (and >= 3 m apart), preferring points near the border of the camera image (cam.box: black bars are outside it) so
    the actor walks in from / out of view (the path continues straight beyond E and B; in a bar the compositor hides
    it, as a real person leaves the image there).
    box_of(X, E) -> rest box (the actor's paths, F1 D33): only exits B whose walk-out passes exit_clear are drawn, and
    when none is left for an entry another entry is tried; on a narrow track the actor walks through in either
    direction or, when both cross the object (it lies ahead on the track), goes back the way it came (B = E).
    Without box_of (the distractors of D33 / D34, before the line library) nothing changes, so E != B.
    st (structure masks, F4 D34): boundary points in view where a walker of `height` m would stand on or behind a
    structure are no entry or exit; for the actor (box_of given) also the spot X (feet and body), the object's rest
    box (obj_h m tall; an entry whose rest box fails is not used) and the in-view path E-X-B with its walk-in /
    walk-out extensions (span s at `speed`: at most the clip, so the plate length; clear) must be free; with carry
    (dict(scale, obj): the actor's scale and carried object, D38) also the crouch at X (structures.crouch_conflict) and
    the carried object on the path and while turning at X (clear). Distractors walk lines of the distractor line
    library (plan_distractors, F3 D36)."""
    poly = np.asarray(cal["walkable"], float)
    lo, hi = poly.min(0), poly.max(0)
    pts = rng.uniform(lo, hi, (tries, 2))
    foot, head, _ = cam.person_px(pts, 1.7)
    rel = (foot[:, 1] - head[:, 1]) / cam.H
    ok = inside(poly, pts) & cam.visible(pts, 1.9, 0.08) & (rel >= MIN_PX) & qc.rest_in_image(cam, pts)
    if obst is not None:  # lit spot, feet not on a parked vehicle (DECISIONS.md D24)
        ok &= np.array([qc.spot_luma(cam, x, obst["luma"]) >= qc.MIN_SPOT_LUMA for x in pts])
        ok &= np.array([qc.feet_on_vehicle(cam, x[None], obst["vehicles"]) == 0 for x in pts])
    actor = box_of is not None
    sx = st if actor else None  # structure tests of the actor's spot, object and path
    if sx is not None:  # standing at X: feet and body off the structures (D34)
        i = np.flatnonzero(ok)
        ok[i] &= ~structures.ground_conflict(cam, sx, pts[i], height)
    ext = span * speed
    order = np.flatnonzero(ok)
    if target is not None and len(order):
        order = order[np.argsort(np.abs(np.log(rel[order] / target)))]
        order = np.r_[rng.permutation(order[:5]), order[5:]]
    bp = boundary_points(poly)
    fb, _, zb = cam.person_px(bp)
    edge = np.where(zb > 0.5, cam.edge(fb), 0.0)
    bok = np.ones(len(bp), bool) if st is None else ~structures.ground_conflict(cam, st, bp, height)
    for X in pts[order][:20]:
        dist = np.linalg.norm(bp - X, axis=1)
        cand = np.flatnonzero((dist >= dmin) & (dist <= dmax) & bok)
        if len(cand) < 2:
            continue
        best = cand[np.argsort(edge[cand])[:max(2, len(cand) // 4)]]
        for _ in range(10):
            E = bp[rng.choice(best)]
            box = None if box_of is None else box_of(X, E)
            if sx is not None and (structures.object_conflict(cam, sx, box, obj_h) or (carry is not None and
                                   structures.crouch_conflict(cam, sx, X, math.atan2(*(X - E)[::-1]), carry["scale"], True))):
                continue  # the object would rest (or the crouch reach) on or behind a structure; another entry turns it
            rest = [i for i in best if np.linalg.norm(bp[i] - E) >= 3.0 and (box is None or exit_clear(X, bp[i], box))]
            if not rest:
                rest = [i for i in cand if np.linalg.norm(bp[i] - E) >= 3.0 and (box is None or exit_clear(X, bp[i], box))]
            if not rest:
                if box is None:
                    break
                continue  # another entry may leave a clear exit
            B = bp[rng.choice(rest)]
            if clear(cam, E, X, B, obst, sx, height, ext, carry):
                return X, E, B
    # narrow track (real people walked a strip shorter than dmin): walk along its main axis through X, from where
    # the person enters the image to where they leave it (at most 12 m each way)
    if len(order):
        c = poly - poly.mean(0)
        u = np.linalg.svd(c, full_matrices=False)[2][0]
        X = pts[order[0]]
        ends = []
        for sgn in (-1.0, 1.0):
            r = np.arange(0.25, 12.01, 0.25)
            line = X + sgn * r[:, None] * u
            vis = cam.visible(line, 1.7, 0.0)
            k = np.argmin(vis) if not vis.all() else len(r) - 1  # first point out of view
            ends.append(line[k])
        if all(np.linalg.norm(e - X) >= 2.0 for e in ends):
            e0, e1 = ends if rng.random() < 0.5 else ends[::-1]
            opts = [(e0, e1)] if box_of is None else [(e0, e1), (e1, e0), (e0, e0), (e1, e1)]  # through, or back
            for E, B in opts:
                if box_of is not None and not exit_clear(X, B, box_of(X, E)):
                    continue
                if sx is not None and (structures.object_conflict(cam, sx, box_of(X, E), obj_h) or (carry is not None and
                                       structures.crouch_conflict(cam, sx, X, math.atan2(*(X - E)[::-1]), carry["scale"], True))):
                    continue
                if clear(cam, E, X, B, obst, sx, height, ext, carry):
                    return X, E, B
    return None


def avatar(rng, t, gender=None):
    """An avatar row, uniform over the table (or over one gender: run_batch keeps a group id's gender over redraws)."""
    av = t.avatars if gender is None else t.avatars[t.avatars.gender == gender]
    return av.sample(1, random_state=int(rng.integers(1 << 31))).iloc[0]


def sample_group(rng, t, gid, families=("place_leave",), neg_weights=None, gender=None, night=None):
    pos_family = rng.choice(families)
    negs = NEG_OF[pos_family]
    w = neg_weights if neg_weights is not None else np.array([NEG_WEIGHTS[n] for n in negs])
    neg_family = rng.choice(negs, p=np.asarray(w) / np.sum(w))
    bend = float(rng.uniform(1.1, 1.7)); stand = float(rng.uniform(1.0, 1.5))
    stop = float(rng.uniform(0.3, 0.8))  # standing still at X before the bend (>= the 0.3 s cross-fade)
    pause = float(rng.uniform(1.0, 4.0)); carry_stand = float(rng.uniform(0.6, 1.6))
    after0 = need_after_event(str(neg_family), bend, stand, pause, carry_stand)  # + the plate's frame step below
    target = float(max(MIN_PX, rng.choice(t.real_rel_h)))  # actor size law (DECISIONS.md D23)
    # actor and object before the path: the planner keeps the walk-out clear of where the object will rest (F1, D33)
    av = avatar(rng, t, gender)
    g = av.gender
    height = float(np.clip(rng.normal(*HEIGHT[g]), HEIGHT[g][0] - 0.15, HEIGHT[g][0] + 0.15))
    scale = height / float(av.height_m)
    walk = str(rng.choice(WALKS[g]))
    v = walk_speed(t, walk, av.avatar, scale)
    oi = rng.choice(len(OBJECT_LAW), p=np.array([o[3] for o in OBJECT_LAW]) / sum(o[3] for o in OBJECT_LAW))
    obj, oscale, ocol = OBJECT_LAW[oi][:3]
    colour = (LIGHT if ocol == "light" else DARK)[int(rng.integers(5))] if ocol else None
    osc, oyaw = float(oscale * rng.uniform(0.9, 1.1)), float(rng.uniform(-30, 30))
    half, obj_h = object_half(obj, osc, t.objects), object_height(obj, osc, t.objects)

    sd = SETDOWN.get(obj, 0.0)

    def box_of(X, E):
        return rest_box(X, math.atan2(*(X - E)[::-1]), g, scale, half, oyaw, setdown=sd)
    for _ in range(50):
        # scene uniform, then plate within the scene; duration law capped by the plate (spec), lengthened only
        # when the outcomes or the stand-ups would not fit (D22, D32)
        ok = t.plates[(t.plates.plate_id.map(t.uses) < MAX_USES) & (t.plates.dur >= 0.8 + after0 + 0.2)]
        if night is not None:  # run_batch decides night per group (NIGHT_SHARE), kept over its redraws
            ok = ok[ok.night == night]
        if ok.empty:
            raise NoPlan("no plate left" + ("" if night is None else f" (night={night})"), (pos_family, neg_family), g)
        scene = rng.choice(t.scenes(ok, "foot"))
        pr = ok[ok.scene_id == scene].sample(1, random_state=int(rng.integers(1 << 31))).iloc[0]
        after = need_after_event(str(neg_family), bend, stand, pause, carry_stand, 1.0 / render_fps(float(pr.fps)))
        d = min(max(float(np.clip(rng.choice(t.real_T), 5.0, 110.0)), 0.8 + after + 0.2), float(pr.end - pr.start) - 1e-3)
        cal = t.calib(pr.plate_id)
        cam = Camera(cal, cal["render_size"])
        st = t.struct(pr.plate_id)
        kw = dict(target=target, obst=t.obstacles.get(pr.plate_id), box_of=box_of, st=st, height=height, speed=v,
                  obj_h=obj_h, span=min(STRUCT_EXT, float(pr.end - pr.start)), carry=dict(scale=scale, obj=(half, obj_h, oyaw)))
        path = plan_path(rng, cal, cam, **kw)
        if path is None:  # the masks can leave too little ground for 4 m entries (indoor lobbies, dump sites; D34)
            path = plan_path(rng, cal, cam, dmin=DMIN_SMALL, **kw)
        if path is not None:
            break
    else:
        raise NoPlan("no plate with a usable path", (pos_family, neg_family), g)
    t.take(pr, "foot")
    X, E, B = path
    # timing: label onset law (50% uniform in [1, d - 2], 50% real g), then the motion starts DELTA earlier; the clip
    # is lengthened within the plate (not the onset moved) so that both outcomes and the stand-ups fit (D32); d in
    # whole ms, so the stored duration is exactly the one t_event was budgeted against
    onset = rng.uniform(1.0, d - 2.0) if rng.random() < 0.5 else float(rng.choice(t.real_g))
    d = min(max(d, onset - delta(str(pos_family)) + after), float(pr.end - pr.start) - 1e-3)
    d = math.floor(d * 1000.0) / 1000.0
    t_event = float(np.clip(onset - delta(str(pos_family)), 0.8, d - after))
    t_arrive = t_event - stop
    t_walk0 = t_arrive - np.linalg.norm(X - E) / v
    # time of day / lighting: inherited from the contest clip's label, MEVA by clock time and luma (< 50: dim)
    night = plate_night(t.clip_meta, pr)
    light = group_light(rng, t, pr, night, plate_dim(t.clip_meta, pr))  # plate_light.csv (D40)
    gr = dict(group_id=gid, families=[str(pos_family), str(neg_family)], plate_id=pr.plate_id, video=pr.video,
              scene_id=pr.scene_id, fold=int(pr.fold), source=pr.source, plate_start=float(rng.uniform(pr.start, pr.end - d)),
              duration=round(d, 3), render_size=cal["render_size"], render_fps=render_fps(float(pr.fps)),
              cam=snapshot(cal), seed=int(rng.integers(1 << 31)),
              actor=dict(avatar=av.avatar, gender=g, scale=scale, height_m=height, walk=walk, speed=v,
                         idle=IDLE[g], crouch_in=CROUCH_IN[g], crouch_out=CROUCH_OUT[g]),
              object=dict(asset=obj, object_type="plastic_bag" if oscale < 1 else t.objects.loc[obj, "object_type"],
                          scale=osc, colour=colour, yaw=oyaw, setdown=sd),
              path=dict(E=E.tolist(), X=X.tolist(), B=B.tolist()), t_event=t_event, t_walk0=float(t_walk0),
              bend=bend, stand=stand, stop=stop, pause=pause, carry_stand=carry_stand,
              light=light, distractors=[], face_blur=bool(rng.random() < FACE_BLUR),
              degrade_seed=int(rng.integers(1 << 31)), structures=dict(st.meta))
    gr["distractors"] = plan_distractors(t, gr, members(gr), cam, walker_screen(gr, cam))
    return gr


def carrying(m, ts):
    """Member m's actor holds the object at times ts (object state hand or grasp)."""
    st = sorted(m["object_states"], key=lambda s: s["t0"])
    i = np.searchsorted([s["t0"] for s in st], np.asarray(ts, float), side="right") - 1
    return np.array([s["state"] in ("hand", "grasp") for s in st])[np.clip(i, 0, len(st) - 1)]


def in_view(cam, xy, height=1.8, width=0.7):
    """A walker at ground points xy [N,2] (a width x height m upright box) overlaps the camera image (cam.box)."""
    foot, head, z = cam.person_px(xy, height)
    half = 0.5 * width * np.abs(foot[:, 1] - head[:, 1]) / height
    x0, y0, x1, y1 = cam.box
    return (z > 0.1) & (foot[:, 0] + half > x0) & (foot[:, 0] - half < x1) & (foot[:, 1] > y0) & (head[:, 1] < y1)


def distractor_conflict(cam, D, ts, X, rc, others):
    """Why distractor D (path, t0, speed, carry) may not join the group, or None (F1, D33): its in-clip line comes
    within DIS_CLEAR m of the spot X or the object's rest centre rc, or its root comes within WALK_CLEAR m
    (WALK_CLEAR_CARRY when either carries an object) of another walker's root at the same time ts while either is in
    view. others = [(track [T,2] at ts, carrying [T] bool)]: every member's actor and the earlier distractors, so the
    decision is the same for all members."""
    keys = walker_keys(D["path"], D["t0"], D["speed"])
    P0, P1 = track(keys, [0.0, float(ts[-1]) + 0.1])  # one straight line (walker_keys): the in-clip part
    if min(seg_point_dist(X, P0, P1), seg_point_dist(rc, P0, P1)) < DIS_CLEAR:
        return "near the spot"
    w = track(keys, ts); vw = in_view(cam, w)
    for o, oc in others:
        lim = np.where(oc | (D.get("carry") is not None), WALK_CLEAR_CARRY, WALK_CLEAR)
        if ((np.linalg.norm(w - o, axis=1) < lim) & (vw | in_view(cam, o))).any():
            return "near a walker"
    return None


def seg_box_dist(P0, P1, box, step=0.02):
    """Distance (m) from the segment P0-P1 to a rest box (centre, angle, half extents; 0 if they meet), to step / 2."""
    c, th, h = box
    n = max(2, int(np.ceil(np.linalg.norm(np.subtract(P1, P0)) / step)) + 1)
    L = (np.asarray(P0, float) + np.linspace(0, 1, n)[:, None] * np.subtract(P1, P0) - c) @ rot(th)
    return float(np.hypot(*np.maximum(np.abs(L) - h, 0.0).T).min())


def distractor_why(cam, D, ts, X, box, others, screen=None):
    """Why distractor D (path, t0, speed, carry) may not join the group, or None (F3, D36): its in-clip line comes
    within DIS_OBJ m of the spot X or of the object's rest box (every member and time alike: the object lies at X
    only in some members); distractor_conflict (WALK_CLEAR / WALK_CLEAR_CARRY root to root from every member's actor
    and the earlier distractors while either is in view; others = [(track at ts, carrying)]); screen(D, track at ts)
    -> the strict screen's reasons for this walker alone (postqc.walker_screen: parked vehicles, structures, moving
    traffic, real people), so a distractor never makes its group fail the screen."""
    keys = walker_keys(D["path"], D["t0"], D["speed"])
    P0, P1 = track(keys, [0.0, float(ts[-1]) + 0.1])  # one straight line: the in-clip part
    if min(seg_point_dist(X, P0, P1), seg_box_dist(P0, P1, box)) < DIS_OBJ:
        return "near the object"
    why = distractor_conflict(cam, D, ts, X, box[0], others)
    if why is None and screen is not None:
        why = (screen(D, track(keys, ts)) or [None])[0]
    return why


def walker_screen(gr, cam):
    """screen(D, track) for plan_distractors: postqc.walker_screen of this group (scripts/postqc.py, imported here
    because it imports this module)."""
    if str(SYN / "scripts") not in sys.path:
        sys.path.insert(0, str(SYN / "scripts"))
    import postqc
    def screen(D, w):
        dc = distractor_carry(D)
        a = math.atan2(*(np.subtract(D["path"][1], D["path"][0]))[::-1])
        return postqc.walker_screen(gr, w, cam, structures.distractor_height(D),
                                    None if dc is None else (np.full(len(w), a), *dc))
    return screen


def plan_distractors(t, gr, ms, cam, screen=None, rng=None, log=None):
    """Passers-by of a group (F3, D36), drawn after its members exist (ms: both outcomes; plan_member never reads the
    distractors, so the members do not change). 0, 1 or 2 (DIS_LAW) on a plate with >= MIN_LINES approved library
    lines (t.lines; none otherwise). Per try: a walker (no clone of the actor or an earlier distractor, D29), an
    approved line in a random direction, the start time so that its feet are in view for >= min(DIS_VIS s, its
    crossing) inside the clip (no distractor is wasted); kept if distractor_why finds nothing (clearance from the spot,
    the rest box, every member's actor and the earlier distractors; screen: the strict screen on its own track). After
    DIS_TRIES failed tries that distractor is left out of the group (both members), never the group failed for it.
    The random stream is its own ((group seed, DIS_SEED)), so the group's other draws do not depend on the library.
    log: list that receives one dict(k, why, D) per try (dry runs, tests)."""
    rng = np.random.default_rng([gr["seed"], DIS_SEED]) if rng is None else rng
    n = int(rng.choice(len(DIS_LAW), p=DIS_LAW))
    lines = t.lines(gr["plate_id"])
    if n == 0 or len(lines) < MIN_LINES:
        return []
    d = gr["duration"]; ts = np.arange(0, d, 0.1)
    X, box = np.array(gr["path"]["X"], float), group_rest_box(gr)
    others = [(track(m["root_keys"], ts), carrying(m, ts)) for m in ms]
    used = [gr["actor"]["avatar"]]
    out = []
    for k in range(n):
        for _ in range(DIS_TRIES):
            dav = t.avatars[~t.avatars.avatar.isin(used)].sample(1, random_state=int(rng.integers(1 << 31))).iloc[0]
            dscale = float(np.clip(rng.normal(*HEIGHT[dav.gender]), *DIS_H)) / float(dav.height_m)
            dwalk = str(rng.choice(WALKS[dav.gender]))
            dv = walk_speed(t, dwalk, dav.avatar, dscale)
            ln = lines[int(rng.integers(len(lines)))]
            E, B, s_in, s_out = ln["E"], ln["B"], ln["s_in"], ln["s_out"]
            if rng.random() < 0.5:  # walked the other way
                L = float(np.linalg.norm(np.subtract(B, E)))
                E, B, s_in, s_out = B, E, L - s_out, L - s_in
            vis = (s_out - s_in) / dv
            need = min(DIS_VIS, vis)
            t_in = float(rng.uniform(need - vis, d - need))  # when the feet enter the image
            D = dict(avatar=dav.avatar, gender=dav.gender, scale=dscale, walk=dwalk, speed=dv,
                     path=[list(map(float, E)), list(map(float, B))], t0=t_in - s_in / dv,
                     carry=str(rng.choice(CARRY_OBJECTS)) if rng.random() < 0.4 else None, line=ln["id"])
            why = distractor_why(cam, D, ts, X, box, others, screen)
            if log is not None:
                log.append(dict(k=k, why=why, D=D))
            if why is None:
                out.append(D); used.append(D["avatar"])
                others.append((track(walker_keys(D["path"], D["t0"], D["speed"]), ts),
                               np.full(len(ts), D["carry"] is not None)))
                break
    return out


def render_fps(fps):
    """min(plate rate, 15 fps); a 25-60 fps plate is rendered at half or less of its rate (DECISIONS.md D17)."""
    for k in (1, 2, 3, 4, 5, 6):
        if fps / k <= 15.5:
            return float(fps / k)
    return 15.0


def col_bottom(vlab, vid):
    """Lowest mask row of vehicle `vid` in every image column (-1 where it has no pixel)."""
    m = vlab == vid
    rows = np.where(m, np.arange(m.shape[0])[:, None], -1)
    return rows.max(0)


def plan_vehicle(rng, t, pr, cam, target, dmin=2.5, dmax=7.0, st=None, height=1.8, speed=1.5, box_of=None, obj_h=0.6,
                 span=STRUCT_EXT, carry=None):
    """Car mode (unload_vehicle_leave, D26/D27): a vehicle still for the whole plate window (plate_vehicles.py); the
    actor's place S on the camera side of it (first ground point from its ground contact towards X whose pixel is
    outside every vehicle mask and below the vehicle's lowest mask row in that column, i.e. in front of it); a spot
    X 2.5-7 m away (walkable, visible, lit, size law, not on a vehicle, rest area out of black bars); for
    carry_from_car an exit B on the walkable polygon's boundary near the border of the camera image (cam.box), >= 3 m
    from X, not back towards the car. S is searched only inside the camera image. Paths avoid vehicle masks.
    st (structure masks, F4 D34): the spot X, the place S (the search goes on past a blocked point), every point of
    S -> X (walked both ways), the object's rest box (box_of(X, S), obj_h m tall) and the exit X -> B with its
    walk-out extension (span s at `speed`, in view) stay off the structures for a walker of `height` m; with carry
    (dict(scale, obj), D38) also the crouch at X and the carried object on S -> X and back, X -> B with its extension
    and the turns at the car and at X (carry_clear; every member's, whatever the negative family)."""
    cal = t.calib(pr.plate_id); vlab = t.vlab(pr.plate_id); obst = t.obstacles.get(pr.plate_id)
    vs = [v for v in t.vehicles.get(pr.plate_id, []) if v["cls"] in (2, 7) and v["contact_uv"][1] < cam.H - 3
          and v.get("n_frames", 0) == v.get("n_of", -1)]  # parked for the whole plate window
    if not vs or vlab is None:
        return None
    poly = np.asarray(cal["walkable"], float)
    pts = rng.uniform(poly.min(0) - 3, poly.max(0) + 3, (600, 2))
    foot, head, _ = cam.person_px(pts, 1.7)
    rel = (foot[:, 1] - head[:, 1]) / cam.H
    ok = inside(poly, pts) & cam.visible(pts, 1.9, 0.05) & (rel >= MIN_PX) & qc.rest_in_image(cam, pts)
    ok &= np.array([qc.spot_luma(cam, x, obst["luma"]) >= qc.MIN_SPOT_LUMA for x in pts])
    ok &= np.array([qc.feet_on_vehicle_mask(cam, x[None], vlab) == 0 for x in pts])
    if st is not None:  # standing at X: feet and body off the structures (D34)
        i = np.flatnonzero(ok)
        ok[i] &= ~structures.ground_conflict(cam, st, pts[i], height)
    bp = boundary_points(poly)
    fb, _, zb = cam.person_px(bp)
    edge = np.where(zb > 0.5, cam.edge(fb), 0.0)
    bok = np.ones(len(bp), bool) if st is None else ~structures.ground_conflict(cam, st, bp, height)
    ext = span * speed
    for v in rng.permutation(vs):
        C = cam.ground(np.array([v["contact_uv"]]))[0]
        if not np.isfinite(C).all():
            continue
        cb = col_bottom(vlab, v["id"])
        dist = np.linalg.norm(pts - C, axis=1)
        cand = np.flatnonzero(ok & (dist >= dmin) & (dist <= dmax))
        if not len(cand):
            continue
        cand = cand[np.argsort(np.abs(np.log(rel[cand] / target)))][:8]
        for i in rng.permutation(cand):
            X = pts[i]; u = (X - C) / np.linalg.norm(X - C)
            S = None
            for r in np.arange(0.6, 2.01, 0.1):
                P = C + r * u
                (pu, pv), = cam.project(np.array([[P[0], P[1], 0.0]]))[0]
                if not (0 <= pu < cam.W and 0 <= pv < cam.H and cam.in_image([[pu, pv]])[0]):
                    break
                in_front = cb[int(pu)] < 0 or pv >= cb[int(pu)] + 2  # feet below the car's lowest row there
                if in_front and qc.feet_on_vehicle_mask(cam, P[None], vlab) == 0 and cam.visible(P[None], 1.7, 0.0)[0] \
                        and (st is None or not structures.ground_conflict(cam, st, P[None], height)[0]):
                    S = P; break
            if S is None or np.linalg.norm(X - S) < 2.0:
                continue
            seg = S + np.linspace(0, 1, 40)[:, None] * (X - S)
            if qc.feet_on_vehicle_mask(cam, seg, vlab) > 0:
                continue
            if st is not None and (any(structures.walk_conflict(cam, st, path_points(S, S, X, 0.0, STRUCT_STEP), height, True))
                                   or (box_of is not None and structures.object_conflict(cam, st, box_of(X, S), obj_h))):
                continue
            a_in, a_car = math.atan2(*(X - S)[::-1]), math.atan2(*(C - S)[::-1])
            if st is not None and carry is not None and (
                    structures.crouch_conflict(cam, st, X, a_in, carry["scale"], True) or not carry_clear(
                        cam, st, [(S, X, 0.0, 0.0), (X, S, 0.0, 0.0)],
                        [(S, a_car, a_in, False), (S, a_in + math.pi, a_car, True), (X, a_in, a_in + math.pi, False)], carry)):
                continue
            exits = [k for k in np.flatnonzero((edge <= 0.12) & bok)
                     if np.linalg.norm(bp[k] - X) >= 3.0 and np.dot(bp[k] - X, X - S) > -0.5 * np.linalg.norm(bp[k] - X) * np.linalg.norm(X - S)]
            B = None
            for k in rng.permutation(exits)[:15]:
                seg_pts = X + np.linspace(0, 1, 40)[:, None] * (bp[k] - X)
                if qc.feet_on_vehicle_mask(cam, path_points(X, X, bp[k]), vlab) == 0 and inside(poly, seg_pts).mean() >= 0.7 \
                        and (st is None or not any(structures.walk_conflict(
                            cam, st, path_points(X, X, bp[k], ext=ext, step=STRUCT_STEP), height, True))) \
                        and (st is None or carry is None or carry_clear(
                            cam, st, [(X, bp[k], 0.0, ext)], [(X, a_in, math.atan2(*(bp[k] - X)[::-1]), False)], carry)):
                    B = bp[k]; break
            if B is None:
                continue
            return X, S, B, C, v["id"]
    return None


def sample_vehicle_group(rng, t, gid, neg_weights=None, gender=None, night=None):
    """One unload_vehicle_leave group: like sample_group, but on a plate with a still parked car, the actor starting
    at the car (D26/D27). Timeline: at the car until t_walk0, turn in place, walk to X at v, stop, bend at t_event."""
    pos_family = "unload_vehicle_leave"
    negs = NEG_OF[pos_family]
    w = neg_weights if neg_weights is not None else np.array([NEG_WEIGHTS[n] for n in negs])
    neg_family = str(rng.choice(negs, p=np.asarray(w) / np.sum(w)))
    bend = float(rng.uniform(1.1, 1.7)); stand = float(rng.uniform(1.0, 1.5)); stop = float(rng.uniform(0.3, 0.8))
    pause = float(rng.uniform(1.0, 4.0)); carry_stand = float(rng.uniform(0.6, 1.6))
    target = float(max(MIN_PX, rng.choice(t.real_rel_h)))
    av = avatar(rng, t, gender)
    g = av.gender
    height = float(np.clip(rng.normal(*HEIGHT[g]), HEIGHT[g][0] - 0.15, HEIGHT[g][0] + 0.15))
    scale = height / float(av.height_m)
    walk = str(rng.choice(WALKS[g]))
    v = walk_speed(t, walk, av.avatar, scale)
    # object before the plate loop: the planner keeps its rest box off the structures (D34)
    oi = rng.choice(len(OBJECT_LAW), p=np.array([o[3] for o in OBJECT_LAW]) / sum(o[3] for o in OBJECT_LAW))
    obj, oscale, ocol = OBJECT_LAW[oi][:3]
    colour = (LIGHT if ocol == "light" else DARK)[int(rng.integers(5))] if ocol else None
    osc, oyaw = float(oscale * rng.uniform(0.9, 1.1)), float(rng.uniform(-30, 30))
    half, obj_h = object_half(obj, osc, t.objects), object_height(obj, osc, t.objects)

    sd = SETDOWN.get(obj, 0.0)

    def box_of(X, S_):
        return rest_box(X, math.atan2(*(X - S_)[::-1]), g, scale, half, oyaw, setdown=sd)
    cand = t.plates[(t.plates.plate_id.map(t.uses) < MAX_USES) & t.plates.plate_id.map(t.still_car)]
    if night is not None:
        cand = cand[cand.night == night]
    if cand.empty:
        raise NoPlan("no car-mode plate left" + ("" if night is None else f" (night={night})"), (pos_family, neg_family), g)
    for _ in range(60):
        scene = rng.choice(t.scenes(cand, "car"))
        pr = cand[cand.scene_id == scene].sample(1, random_state=int(rng.integers(1 << 31))).iloc[0]
        cal = t.calib(pr.plate_id); cam = Camera(cal, cal["render_size"])
        st = t.struct(pr.plate_id)
        res = plan_vehicle(rng, t, pr, cam, target, st=st, height=height, speed=v, box_of=box_of, obj_h=obj_h,
                           span=min(STRUCT_EXT, float(pr.end - pr.start)), carry=dict(scale=scale, obj=(half, obj_h, oyaw)))
        if res is None:
            continue
        X, S_, B, C, vid = res
        L = float(np.linalg.norm(X - S_))
        yaw_car = math.atan2(*(C - S_)[::-1]); a_in = math.atan2(*(X - S_)[::-1])
        walk_t = turn_time(yaw_car, a_in) + L / v  # turn in place, then walk (plan_member)
        back_t = turn_time(a_in, a_in + math.pi) + L / v + 1.0  # turn at X, walk back, turn to the car
        need = vehicle_need_after_event(neg_family, bend, stand, pause, carry_stand, back_t,
                                        1.0 / render_fps(float(pr.fps)))
        # both members have turned at X and walked POS_UP s (carry exits to B, the others turn back to the car; D35)
        tts = [turn_time(a_in, math.atan2(*(B - X)[::-1])) if OUTCOME[f] == "carry" else turn_time(a_in, a_in + math.pi)
               for f in (pos_family, neg_family)]
        need = max(need, walkout_after_event((pos_family, neg_family), bend, stand, pause, carry_stand, tts,
                                             1.0 / render_fps(float(pr.fps))))
        plate_len = float(pr.end - pr.start) - 1e-3
        lead = walk_t + stop  # t_walk0 -> t_event
        d = min(max(float(np.clip(rng.choice(t.real_T), 5.0, 110.0)), 0.5 + lead + need + 0.3), plate_len)
        t_max = d - lead - need  # latest t_walk0
        if t_max < 0.5:
            continue
        # label onset = t_walk0 + delta: same event-time law as sample_group (50% uniform, 50% real g), clipped
        onset = rng.uniform(1.0, max(1.0, t_max + delta(pos_family))) if rng.random() < 0.5 else float(rng.choice(t.real_g))
        t_leave = float(np.clip(onset - delta(pos_family), 0.5, t_max))
        te = t_leave + lead
        break
    else:
        raise NoPlan("no plate with a usable parked car", (pos_family, neg_family), g)
    t.take(pr, "car")
    night = plate_night(t.clip_meta, pr)
    light = group_light(rng, t, pr, night, plate_dim(t.clip_meta, pr))  # plate_light.csv (D40)
    gr = dict(group_id=gid, families=[pos_family, neg_family], plate_id=pr.plate_id, video=pr.video,
              scene_id=pr.scene_id, fold=int(pr.fold), source=pr.source,
              plate_start=float(rng.uniform(pr.start, pr.end - d)), duration=round(d, 3),
              render_size=cal["render_size"], render_fps=render_fps(float(pr.fps)), cam=snapshot(cal),
              seed=int(rng.integers(1 << 31)),
              actor=dict(avatar=av.avatar, gender=g, scale=scale, height_m=height, walk=walk, speed=v,
                         idle=IDLE[g], crouch_in=CROUCH_IN[g], crouch_out=CROUCH_OUT[g]),
              object=dict(asset=obj, object_type="plastic_bag" if oscale < 1 else t.objects.loc[obj, "object_type"],
                          scale=osc, colour=colour, yaw=oyaw, setdown=sd),
              path=dict(E=S_.tolist(), X=X.tolist(), B=B.tolist()), t_event=float(te), t_walk0=float(t_leave),
              bend=bend, stand=stand, stop=stop, pause=pause, carry_stand=carry_stand, light=light, distractors=[],
              face_blur=bool(rng.random() < FACE_BLUR), degrade_seed=int(rng.integers(1 << 31)),
              vehicle=dict(id=int(vid), contact=C.tolist(), yaw_car=float(yaw_car)), structures=dict(st.meta))
    # distractors as on foot (D38, review R1: car mode never planned them; the library was not the reason)
    gr["distractors"] = plan_distractors(t, gr, members(gr), cam, walker_screen(gr, cam))
    return gr


def upright_after_event(neg_family, bend, stand, pause, step=0.0):
    """Seconds after t_event until every member of the group has stood up for the last time, plus POS_UP, plus one
    frame step (1 / render fps: the rendered frames end up to one step before the duration, so the clip's frames
    then still span POS_UP s after the stand-up): the positive's (and pre-existing pick-up's) stand-up ends at bend +
    stand, the pick-up's second one (place_pickup, unload_reload) at 2 bend + 2 stand + pause; carry never crouches.
    Budgeting one label only would leave mid-crouch endings in the other (F5 review blocker, DECISIONS.md D32)."""
    up = 2 * bend + 2 * stand + pause if OUTCOME[neg_family] == "pickup" else bend + stand
    return up + POS_UP + step


def walkout_after_event(families, bend, stand, pause, carry_stand, turns, step=0.0):
    """Seconds after t_event until every member of the group has turned at X and walked POS_UP s of its walk-out, plus
    one frame step (D35): the members' t_walk_out - t_event as plan_member makes it (leave and pre-existing: bend +
    stand + 0.3 s idle + turn; pick-up: 2 bend + 2 stand + pause + 0.3 + turn; carry: carry_stand + turn), turns =
    each member's turn_time. The max over the group's members, so both members get the same clip (no label cue).
    Car mode only: on foot it discarded 7% of the planned paths (short plates) and removed the real short tail after
    the onset (clips ending < 3 s after it 29% -> 1%, real 23%), so on-foot clips may still end before the walk-out."""
    up = {"leave": bend + stand + 0.3, "preexisting": bend + stand + 0.3, "pickup": 2 * bend + 2 * stand + pause + 0.3,
          "carry": carry_stand}
    return max(up[OUTCOME[f]] + tt for f, tt in zip(families, turns)) + POS_UP + step


def need_after_event(neg_family, bend, stand, pause, carry_stand, step=0.0):
    """Seconds needed after t_event so that both members' outcomes happen inside the clip (the negative's outcome
    >= 1 s before the end: pickup's re-grasp, pre-existing grasp, carry's walk-off; at least 2 s) and every member is
    upright POS_UP s before the end (upright_after_event; step = 1 / render fps)."""
    return max(2.0, {"place_pickup": 2 * bend + stand + pause, "pickup_preexisting": bend,
                     "carry_through": carry_stand, "unload_reload": 2 * bend + stand + pause,
                     "load_preexisting": bend, "carry_from_car": carry_stand}.get(neg_family, 0.0) + 1.0,
               upright_after_event(neg_family, bend, stand, pause, step))


def vehicle_need_after_event(neg_family, bend, stand, pause, carry_stand, back_t, step=0.0):
    """Car mode: seconds needed after t_event for the negative's outcome incl. the walk back to the car (back_t) + 1 s,
    for the positive's walk back to the car (back_t) + 1 s in every group (D38, review S1: with carry_from_car only the
    carry's walk-off was budgeted, so a positive could end next to the dropped object, which no car-mode negative
    does), and for every member to be upright POS_UP s before the end (upright_after_event; the outcome budget
    already leaves more, D32). sample_vehicle_group adds walkout_after_event (D35)."""
    need = max({"unload_reload": 2 * bend + stand + pause + 0.3 + back_t, "load_preexisting": bend + stand + 0.3 + back_t,
                "carry_from_car": carry_stand + 1.0}[neg_family], bend + 1.0, bend + stand + 0.3 + back_t) + 1.0
    return max(need, upright_after_event(neg_family, bend, stand, pause, step))


def turn_time(a, b):
    d = abs((b - a + math.pi) % (2 * math.pi) - math.pi)
    return 0.3 + 0.7 * d / math.pi


def plan_member(gr, family):
    """Motion segments, root keys, object states, events and labels of one member."""
    out = OUTCOME[family]
    A = gr["actor"]
    E, X, B = (np.array(gr["path"][k]) for k in "EXB")
    a_in = math.atan2(*(X - E)[::-1]); a_out = math.atan2(*(B - X)[::-1])
    v, d = A["speed"], gr["duration"]
    te = gr["t_event"]; ta = te - gr["stop"]; t0 = gr["t_walk0"]
    u_in = (X - E) / np.linalg.norm(X - E)
    car = gr.get("vehicle")  # car mode: E is the actor's place at the car's near side (DECISIONS.md D26, D27)
    if car is None:
        seg = [dict(anim=A["walk"], t0=t0 - 60.0, t1=ta, mode="loop"), dict(anim=A["idle"], t0=ta, t1=te, mode="once")]
        keys = [(t0 - 60.0, *(E - u_in * v * 60.0), a_in), (t0, *E, a_in), (ta, *X, a_in)]  # walks in from off-view
    else:  # at the car (facing it, holding the object), turns in place at t_walk0, then walks to X at speed v
        a_car = car["yaw_car"]; tt0 = turn_time(a_car, a_in)
        seg = [dict(anim=A["idle"], t0=-60.0, t1=t0 + tt0, mode="loop"), dict(anim=A["walk"], t0=t0 + tt0, t1=ta, mode="loop"),
               dict(anim=A["idle"], t0=ta, t1=te, mode="once")]
        keys = [(-60.0, *E, a_car), (t0, *E, a_car), (t0 + tt0, *E, a_in), (ta, *X, a_in)]
    back = car is not None and out in ("leave", "pickup", "preexisting")
    if back:  # leave, re-load and pre-existing return to the car
        B = E.copy()
        a_out = math.atan2(*(B - X)[::-1])
    ev = dict(t_walk0=t0, t_arrive=ta, t_event=te)
    t_rel = te + gr["bend"]

    def bend_stand(tb):
        """crouch_in over gr['bend'] s from tb, crouch_out over gr['stand'] s; returns end time."""
        seg.append(dict(anim=A["crouch_in"], t0=tb, t1=tb + gr["bend"], mode="once", fit=gr["bend"]))
        seg.append(dict(anim=A["crouch_out"], t0=tb + gr["bend"], t1=tb + gr["bend"] + gr["stand"], mode="once",
                        fit=gr["stand"]))
        return tb + gr["bend"] + gr["stand"]

    def walk_out(tw):
        """Turn in place at X (idle) towards B, then walk at speed v (no walking while turning: D27). A 180 deg turn
        (back the way it came: car mode, narrow tracks) is clockwise, to the side of the object (F1, D33: counter-
        clockwise swept male legs through the resting object in 7 of 82 car-mode returns, clockwise in 0)."""
        tt = turn_time(a_in, a_out)
        dist = np.linalg.norm(B - X)
        back180 = abs((a_out - a_in + math.pi) % (2 * math.pi) - math.pi) > math.pi - 1e-6
        keys.append((tw, *X, a_in))
        keys.append((tw + tt, *X, a_in - math.pi + 1e-9 if back180 else a_out))
        seg.append(dict(anim=A["idle"], t0=tw, t1=tw + tt, mode="once"))
        tb = tw + tt + dist / v
        keys.append((tb, *B, a_out))
        if back:  # at the car again: turn to it in place and stay
            seg.append(dict(anim=A["walk"], t0=tw + tt, t1=tb, mode="loop"))
            seg.append(dict(anim=A["idle"], t0=tb, t1=d + 5.0, mode="loop"))
            keys.append((tb + turn_time(a_out, car["yaw_car"]), *B, car["yaw_car"]))
            ev["t_back"] = tb
        else:
            seg.append(dict(anim=A["walk"], t0=tw + tt, t1=d + 5.0, mode="loop"))
            keys.append((tb + 60.0, *(B + (B - X) / dist * v * 60.0), a_out))  # walks on off-view
        return tw + tt

    obj = []
    if out in ("leave", "pickup", "preexisting"):
        t_up = bend_stand(te)
        keys.append((t_up, *X, a_in))
        ev.update(t_release=t_rel, t_stand=t_up)
        if out == "leave":
            seg.append(dict(anim=A["idle"], t0=t_up, t1=t_up + 0.3, mode="once"))
            tw = walk_out(t_up + 0.3)
            obj = [dict(state="hand", t0=-1e9), dict(state="drop", t0=t_rel)]
        elif out == "pickup":
            tb2 = t_up + gr["pause"]
            seg.append(dict(anim=A["idle"], t0=t_up, t1=tb2, mode="once"))
            keys.append((tb2, *X, a_in))
            t_grasp = tb2 + gr["bend"]
            t_up2 = bend_stand(tb2)
            seg.append(dict(anim=A["idle"], t0=t_up2, t1=t_up2 + 0.3, mode="once"))
            tw = walk_out(t_up2 + 0.3)
            obj = [dict(state="hand", t0=-1e9), dict(state="drop", t0=t_rel), dict(state="grasp", t0=t_grasp)]
            ev.update(t_grasp=t_grasp, t_pause=t_grasp - t_rel)
        else:  # preexisting: same body motion as leave, object starts on the ground at the drop point
            seg.append(dict(anim=A["idle"], t0=t_up, t1=t_up + 0.3, mode="once"))
            tw = walk_out(t_up + 0.3)
            obj = [dict(state="ground_at_release", t0=-1e9), dict(state="grasp", t0=t_rel)]
            ev.update(t_grasp=t_rel)
    else:  # carry: stands at X instead of bending, then walks out carrying
        tw = te + gr["carry_stand"]
        seg[-1]["t1"] = tw
        keys.append((tw, *X, a_in))
        tw = walk_out(tw)
        obj = [dict(state="hand", t0=-1e9)]
    ev["t_walk_out"] = tw
    # (no 'stow': an object made to vanish next to the car only in negatives was a label cue, D27)
    # t_dec: object back in hand, or the actor >= 2 m from the spot (walking at v from tw, after the turn) for 1 s
    t_dec = ev["t_grasp"] if out in ("pickup", "preexisting") else tw + 2.0 / v + 1.0
    t_dec = t_dec if t_dec <= d else None
    div = {"leave": None, "pickup": ev.get("t_stand", te) + 0.3, "carry": te, "preexisting": 0.0}[out]
    y = int(out == "leave")
    # label onset: start of the disposal motion (+ delta); for unloading the annotators' onset is the object leaving
    # the vehicle (taxonomy), here the start of the walk from the car
    t_on = ((t0 if car is not None else te) + delta(gr["families"][0])) if y else None
    return dict(family=family, outcome=out, y=y, segments=seg, root_keys=[list(map(float, k)) for k in keys],
                object_states=obj, events={k: float(v_) for k, v_ in ev.items()}, t_dec=t_dec,
                t_onset_exact=t_on, timestamp=quantize(t_on) if y else -1, divergence=div)


def members(gr):
    return [plan_member(gr, f) for f in gr["families"]]


def main(a):
    rng = np.random.default_rng(a.seed)
    t = Tables(light_fallback=a.light_fallback)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for k in range(a.groups):
        gid = f"g{a.seed:03d}{k:04d}"
        gr = (sample_vehicle_group(rng, t, gid) if "unload_vehicle_leave" in a.families and
              (a.families == ["unload_vehicle_leave"] or rng.random() < 0.5)
              else sample_group(rng, t, gid, tuple(f for f in a.families if f != "unload_vehicle_leave")))
        for m in members(gr):
            (out / f"{gid}_{m['outcome']}.json").write_text(json.dumps(dict(group=gr, member=m), indent=1))
        print(gid, gr["plate_id"], gr["families"], f"d={gr['duration']:.1f} te={gr['t_event']:.1f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--groups", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--families", nargs="+", default=["place_leave"])
    p.add_argument("--out", default=str(SYN / "data/scenarios"))
    p.add_argument("--light-fallback", action="store_true",
                   help="also sample plates without a plate_light.csv row (old random light for them)")
    main(p.parse_args())
