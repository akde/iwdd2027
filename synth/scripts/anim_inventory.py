"""Stage 3: animation inventory (run inside Blender):
blender -b -P synth/scripts/anim_inventory.py   -> synth/data/anim_inventory.csv
name, folder, frames, fps, dur_s, drift_m (root horizontal start->end), speed_mps, root_z0, root_min_z, root_max_z,
category, sub. Root motion is on the Bip01 object channels of each animation FBX (Rocketbox: 30 fps)."""
import sys, pathlib, csv
import bpy
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "gen"))
import rb


def category(n):
    if "hold_bag" in n or "trolley" in n:
        return "walk-carry", "upper-body carry pose (static)"
    if "crouch_in" in n:
        return "place", "bend down to the ground (squat)"
    if "crouch_out" in n:
        return "pick-up", "stand up from the ground (squat)"
    if "crouch" in n:
        return "idle", "crouched"
    if "_walk" in n and not any(k in n for k in ("to_run", "start", "stop")):
        return "walk", "cycle"
    if any(k in n for k in ("walk_start", "walk_stop", "_to_walk", "run_to_walk", "walk_to_run")):
        return "walk", "transition"
    if "_run" in n:
        return "other", "run"
    if "turn_" in n:
        return "other", "turn in place"
    if "idle" in n or "waiting" in n or "breathe" in n:
        return "idle", ""
    if "throw" in n:
        return "throw", ""
    return "other", ""


bpy.ops.wm.read_factory_settings(use_empty=True)
rows = []
for d in rb.ANIM_DIRS:
    for p in sorted((rb.RB / "Animations" / d).glob("*.max.fbx")):
        name = p.name[:-len(".max.fbx")]
        if any(r["name"] == name for r in rows):
            continue  # xyz duplicates of xy clips
        act = rb.import_action(name)
        loc = {c.array_index: c for c in act.fcurves if c.data_path == "location"}
        f0, f1 = (int(round(x)) for x in act.frame_range)
        xy = [(loc[0].evaluate(f), loc[1].evaluate(f)) for f in (f0, f1)] if loc else [(0, 0), (0, 0)]
        zs = [loc[2].evaluate(f) for f in range(f0, f1 + 1)] if loc else [0.0]
        drift = ((xy[1][0] - xy[0][0]) ** 2 + (xy[1][1] - xy[0][1]) ** 2) ** 0.5
        dur = (f1 - f0) / 30.0
        cat, sub = category(name)
        rows.append(dict(name=name, folder=d, frames=f1 - f0 + 1, fps=30, dur_s=round(dur, 3), drift_m=round(drift, 3),
                         speed_mps=round(drift / dur, 3) if dur else 0.0, root_z0=round(zs[0], 3),
                         root_min_z=round(min(zs), 3), root_max_z=round(max(zs), 3), category=cat, sub=sub))
        bpy.data.actions.remove(act)
out = pathlib.Path(__file__).resolve().parents[1] / "data/anim_inventory.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("INVENTORY", len(rows), out)
