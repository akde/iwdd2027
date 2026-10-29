"""Re-apply the obstacle and darkness QC (qc.feet_on_vehicle, qc.spot_luma) to an existing batch directory.
python synth/scripts/postqc.py [BATCH_DIR=synth/data]
A group is rejected if any walker's feet (actor or distractor, every 0.1 s over the clip, in view) land on a parked
vehicle, or the event spot's plate luma < MIN_SPOT_LUMA. Rejected clips/labels move to BATCH_DIR/rejected/; the
manifest's qc / qc_why are updated. check() is also run_batch's pre-render screen (strict: also outside image,
traffic motion, real person, object on vehicle and the plate structures, 'structure', D31-D34, and every distractor on
an approved line of the distractor line library with its clearance, D36). walker_screen() is the same screen for one
walker alone: plan_distractors tests each distractor with it (F3, D36)."""
import json, math, pathlib, shutil, sys
import cv2, numpy as np, pandas as pd
GEN = pathlib.Path(__file__).resolve().parents[1] / "gen"
sys.path[:0] = [str(GEN), str(GEN.parent / "scripts")]
import qc, composite, degrade, plate_motion, scenario, structures
from camera import Camera, group_calib
from tracks import heading, moving, track, walker_keys, walkers  # ground tracks of the actor and the distractors, as rendered

SYN = GEN.parent
VEHICLE_MASKS = "--masks" in sys.argv  # batches rendered with occlusion (after release v1)
_REGIONS = {}  # (plate, render size, image bottom) -> qc.motion_regions; "objects" -> object_table.csv


def traffic_data(gr):
    """(plate_traffic.py boxes, still parked vehicle boxes normalised) of the group's plate; None without a file."""
    f = SYN / f"data/traffic/{gr['plate_id']}.npz"
    if not f.exists():
        return None
    W, H = gr["render_size"]
    if "vehicles" not in _REGIONS:
        _REGIONS["vehicles"] = json.loads((SYN / "data/vehicles.json").read_text())
    vj = _REGIONS["vehicles"].get(gr["plate_id"], [])
    return np.load(f), [[b / s for b, s in zip(v["box"], (W, H, W, H))] for v in vj
                        if v.get("n_frames", 0) == v.get("n_of", -1)]


def traffic_hit(gr, ms, cam, oxy=None):
    """Moving traffic over a walker's feet or the resting object at oxy (default X; strict: its rest centre, F1 D33)
    (qc.traffic_conflict)."""
    oxy = gr["path"]["X"] if oxy is None else oxy
    td = traffic_data(gr)
    if td is None:
        return False
    tr, still = td
    ts = np.arange(0, gr["duration"], 0.1)
    for m in ms:
        t_obj = m["events"].get("t_release") if m["outcome"] in ("leave", "pickup") else (0.0 if m["outcome"] == "preexisting" else None)
        if qc.traffic_conflict(gr, walkers(gr, m), ts, cam, tr, still, oxy if t_obj is not None else None, t_obj):
            return True
    return False


def regions(gr, cam):
    """qc.motion_regions of the group's plate (cached), None if it has no motion file (plate_motion.py)."""
    key = (gr["plate_id"], tuple(gr["render_size"]), float(cam.box[3]))
    if key not in _REGIONS:
        mf, tf = plate_motion.load(gr["plate_id"]), SYN / f"data/traffic/{gr['plate_id']}.npz"
        _REGIONS[key] = None if mf is None else qc.motion_regions(mf, np.load(tf) if tf.exists() else None,
                                                                   gr["render_size"], cam.box[3])
    return _REGIONS[key]


def motion_hit(gr, ms, cam):
    """A moving vehicle from the plate's motion foreground meets a walker or the resting object (at its rest centre,
    scenario.group_rest_box; qc.motion_conflict; regions attributed to vehicles, qc.motion_regions). None if the plate
    has no motion file (plate_motion.py). Strict only."""
    reg = regions(gr, cam)
    if reg is None:
        return None
    ts = np.arange(0, gr["duration"], 0.1)
    if "objects" not in _REGIONS:
        _REGIONS["objects"] = pd.read_csv(SYN / "data/object_table.csv").set_index("asset")
    o = _REGIONS["objects"].loc[gr["object"]["asset"]]
    r = 0.5 * max(o.dx, o.dy) * gr["object"].get("scale", 1.0)
    rc = scenario.group_rest_box(gr, _REGIONS["objects"])[0]
    for m in ms:
        ev, out = m["events"], m["outcome"]
        t_obj = ev.get("t_release") if out in ("leave", "pickup") else (0.0 if out == "preexisting" else None)
        t_end = ev.get("t_grasp", 1e9) if out in ("pickup", "preexisting") else 1e9
        obj = None if t_obj is None else (rc, r, (ts >= t_obj) & (ts < t_end))
        if qc.motion_conflict(gr, walkers(gr, m), ts, cam, reg, obj):
            return True
    return False


def outside_image(gr, cam):
    """The actor standing at the spot X (1.9 m; in car mode also at its place S at the car, 1.7 m) is not wholly
    inside the camera image (cam.box, black bars outside), or the object's rest area reaches into a bar
    (qc.rest_in_image). The samplers never draw such a spot; this guards the screen (strict only)."""
    X = np.array([gr["path"]["X"]])
    bad = not cam.visible(X, 1.9, 0.0)[0] or not qc.rest_in_image(cam, X)[0]
    return bad or ("vehicle" in gr and not cam.visible(np.array([gr["path"]["E"]]), 1.7, 0.0)[0])


def structure_hit(gr, ms, cam, box=None):
    """Plate structures (F4, D34; gen/structures.py) against every walker of every member every 0.1 s while in view
    (feet hidden; body box over the front layer F or behind a standing structure S, also with the feet below the
    frame), every carried object while held (the actor's in each member, a distractor's), the crouch at the spot
    (D38) and the object's rest box (default scenario.group_rest_box; footprint and projected 3D box): the reasons
    of structures.group_conflict ([] = clear), None if the plate has no structure raster (fail-closed)."""
    st = structures.load(gr["plate_id"])
    if st is None:
        return None
    ts = np.arange(0, gr["duration"], 0.1)
    tracks = [(f"{m['outcome']} actor", track(m["root_keys"], ts)) for m in ms]  # distractors: the same in all members
    tracks += [("distractor", track(walker_keys(D["path"], D["t0"], D["speed"]), ts)) for D in gr["distractors"]]
    heights = [gr["actor"]["height_m"]] * len(ms) + [structures.distractor_height(D) for D in gr["distractors"]]
    box = scenario.group_rest_box(gr) if box is None else box
    o, A = gr["object"], gr["actor"]
    carried = [(f"{m['outcome']} actor", w, heading(m["root_keys"], ts), scenario.carrying(m, ts), A["scale"],
                scenario.carried_obj(gr), moving(m["root_keys"], ts, structures.CARRY_SETTLE)) for m, (_, w) in zip(ms, tracks)]
    for D, (_, w) in zip(gr["distractors"], tracks[len(ms):]):
        if D.get("carry"):
            carried.append(("distractor", w, heading(walker_keys(D["path"], D["t0"], D["speed"]), ts),
                            np.ones(len(ts), bool), *scenario.distractor_carry(D), True))
    E, X = np.array(gr["path"]["E"], float), np.array(gr["path"]["X"], float)
    crouch = (X, math.atan2(*(X - E)[::-1]), A["scale"]) if A.get("crouch_in", "m_crouch_in") == "m_crouch_in" else None
    return structures.group_conflict(cam, st, tracks, heights, (box, scenario.object_height(o["asset"], o.get("scale", 1.0))),
                                     carried, crouch)


def vehicle_labels(plate_id):
    """The plate's parked-vehicle label image (plate_vehicles.py), cached; None without a file."""
    key = ("vlab", plate_id)
    if key not in _REGIONS:
        vf = SYN / f"data/vehicles/{plate_id}.png"
        _REGIONS[key] = cv2.imread(str(vf), cv2.IMREAD_UNCHANGED) if vf.exists() else None
    return _REGIONS[key]


def walker_screen(gr, w, cam, height, carry=None):
    """The strict screen's walker reasons for one extra walker alone (a distractor try, F3 D36): ground track w [T,2]
    at the screen's 0.1 s steps over the clip, `height` m tall; carry = (heading [T], scale, carry_box obj spec) when
    it carries an object (D38). 'on vehicle' (feet on a parked-vehicle mask), 'structure' (structures.walk_conflict
    and the carried object, structures.carry_conflict; also without a raster: fail-closed), 'traffic'
    (qc.traffic_conflict), 'traffic motion' (qc.motion_conflict; also without motion data), 'real person overlap'
    (extended plates, qc.person_overlap). check() tests every walker of the group with the same functions."""
    why, ts = [], np.arange(0, gr["duration"], 0.1)
    if qc.feet_on_vehicle_mask(cam, w, vehicle_labels(gr["plate_id"])) > 0:
        why.append("on vehicle")
    st = structures.load(gr["plate_id"])
    if st is None or any(structures.walk_conflict(cam, st, w, height)) or \
            (carry is not None and structures.carry_conflict(cam, st, w, *carry).any()):
        why.append("structure")
    td = traffic_data(gr)
    if td is not None and qc.traffic_conflict(gr, [w], ts, cam, *td):
        why.append("traffic")
    reg = regions(gr, cam)
    if reg is None or qc.motion_conflict(gr, [w], ts, cam, reg):
        why.append("traffic motion")
    pf = SYN / f"data/det_xe/{gr['plate_id'][:-1]}.npz"
    if gr["plate_id"].split("_")[-1].startswith("x") and pf.exists() and qc.person_overlap(gr, [w], ts, cam, np.load(pf)):
        why.append("real person overlap")
    return why


def distractor_hit(gr, ms, cam):
    """Strict (F3, D36): every distractor walks an approved line of the plate's distractor line library (its id and
    its two ends; scenario.approved_lines with the group's camera and the plate's structure masks) -> 'distractor not
    approved', and keeps plan_distractors' clearance from the spot, the rest box, every member's actor and the
    earlier distractors (scenario.distractor_why without the walker screen, which check() applies to every walker)
    -> 'distractor clearance'. plan_distractors never draws either; this guards the screen."""
    why = set()
    lines = {ln["id"]: ln for ln in scenario.approved_lines(scenario.load_lines(), gr["plate_id"], group_calib(gr),
                                                             structures.load(gr["plate_id"]))}
    ts = np.arange(0, gr["duration"], 0.1)
    X, box = np.array(gr["path"]["X"], float), scenario.group_rest_box(gr)
    others = [(track(m["root_keys"], ts), scenario.carrying(m, ts)) for m in ms]
    for D in gr["distractors"]:
        ln = lines.get(D.get("line"))
        P = np.array(D["path"], float)
        if ln is None or not (np.allclose(P, [ln["E"], ln["B"]]) or np.allclose(P, [ln["B"], ln["E"]])):
            why.add("distractor not approved")
        if scenario.distractor_why(cam, D, ts, X, box, others) is not None:
            why.add("distractor clearance")
        others.append((track(walker_keys(D["path"], D["t0"], D["speed"]), ts), np.full(len(ts), D.get("carry") is not None)))
    return sorted(why)


def check(gr, ms, obst, bd=None, plates=None, vehicle_masks=False, strict=False):
    """Reasons to reject the group (empty: keep). vehicle_masks: feet tested against the parked-vehicle masks
    (plate_vehicles.py; needed with occlusion and the car mode, where the actor stands next to a car) instead of their
    boxes (release v1). strict: also the reasons added for new batches after release v2 (synth/PLAN_D29_FIXES.md),
    passed only by run_batch's pre-render screen; main() never passes it, so re-running it on an old batch
    directory keeps that batch's verdicts. The camera is the one the group was sampled with (camera.group_calib)."""
    cal = group_calib(gr)
    cam = Camera(cal, gr["render_size"]); o = obst[gr["plate_id"]]
    why = []
    box = scenario.group_rest_box(gr) if strict else None  # where the object rests (F1, D33); old batches: X
    if vehicle_masks:
        vf = SYN / f"data/vehicles/{gr['plate_id']}.png"
        vlab = cv2.imread(str(vf), cv2.IMREAD_UNCHANGED) if vf.exists() else None
        on = max(qc.feet_on_vehicle_mask(cam, w, vlab) for m in ms for w in walkers(gr, m))
        if strict:  # the object's rest box (centre and corners) on or behind a parked vehicle
            c, th, h = box
            P = c + np.array([[0, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]]) * h @ scenario.rot(th).T
            if qc.feet_on_vehicle_mask(cam, P, vlab) > 0:
                why.append("object on vehicle")
    else:
        on = max(qc.feet_on_vehicle(cam, w, o["vehicles"]) for m in ms for w in walkers(gr, m))
    if on > 0:
        why.append("on vehicle")
    if qc.spot_luma(cam, gr["path"]["X"], o["luma"]) < qc.MIN_SPOT_LUMA:
        why.append("too dark")
    if strict and outside_image(gr, cam):  # F2 (D31)
        why.append("outside image")
    if strict and gr["distractors"]:  # F3 (D36): approved lines and clearance
        why += distractor_hit(gr, ms, cam)
    if strict:  # F4 (D34): walkers and the object on or behind plate structures
        sh = structure_hit(gr, ms, cam, box)
        if sh is None:
            why.append("no structure mask")  # fail-closed
        elif sh:
            why.append("structure")
    if traffic_hit(gr, ms, cam, None if box is None else box[0]):
        why.append("traffic")
    elif strict:  # F5 (D32): the YOLO box rule OR vehicle-attributed motion (the union never passes what it rejects)
        mh = motion_hit(gr, ms, cam)
        if mh is None:
            why.append("no motion data")  # fail-closed
        elif mh:
            why.append("traffic motion")
    if strict and not qc.person_free(gr["video"], gr["plate_id"], gr["plate_start"], gr["plate_start"] + gr["duration"],
                                     cal):  # before rendering (it was checked only after rendering)
        why.append("real person")
    pf = SYN / f"data/det_xe/{gr['plate_id'][:-1]}.npz"  # extended plates (extend_plates.py): <video>_x<k><letter>
    if gr["plate_id"].split("_")[-1].startswith("x") and pf.exists():
        ts = np.arange(0, gr["duration"], 0.1)
        if any(qc.person_overlap(gr, walkers(gr, m), ts, cam, np.load(pf)) for m in ms):
            why.append("real person overlap")
    if bd is not None:  # actor visibility during the event (needs the clean composite and the masks)
        pos = next(m for m in ms if m["y"] == 1)
        mk = np.load(bd / f"renders/{gr['group_id']}/{pos['outcome']}/masks.npz")
        ks = qc.event_frames(mk, pos, gr["render_fps"])
        fr, _ = degrade.decode(bd / f"clean/{gr['group_id']}_{pos['outcome']}.mp4")
        pl = composite.plate_frames(plates.loc[gr["plate_id"], "path"], gr["plate_start"],
                                    [k / gr["render_fps"] for k in ks], gr["render_size"])
        if qc.visibility(fr, pl, mk["act_box"], ks) < (qc.MIN_VISIBILITY if strict else qc.MIN_VISIBILITY_OLD):
            why.append("actor not visible")
    return why


def main(bd):
    bd = pathlib.Path(bd)
    obst = json.loads((SYN / "data/obstacles.json").read_text())
    plates = pd.read_csv(SYN / "data/plates.csv").set_index("plate_id")
    man = pd.read_csv(bd / "synth_manifest.csv")
    rej = bd / "rejected"; rej.mkdir(exist_ok=True)
    n = 0
    for pid, g in man[man.qc].groupby("pair_id"):
        S = [json.loads((bd / f"scenarios/{c}.json").read_text()) for c in g["clip"]]
        why = check(S[0]["group"], [s["member"] for s in S], obst, bd, plates, vehicle_masks=VEHICLE_MASKS)
        if why:
            n += 1
            man.loc[man.pair_id == pid, "qc"] = False
            man.loc[man.pair_id == pid, "qc_why"] = ";".join(why)
            for c in g["clip"]:
                for sub, ext in (("clips", "mp4"), ("labels", "json")):
                    f = bd / f"{sub}/{c}.{ext}"
                    if f.exists():
                        shutil.move(str(f), str(rej / f.name))
            print(pid, why)
    man.to_csv(bd / "synth_manifest.csv", index=False)
    print(f"post-QC rejected {n} groups; accepted now {man[man.qc].pair_id.nunique()} groups, {int(man.qc.sum())} clips")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--masks"]
    main(args[0] if args else SYN / "data")
