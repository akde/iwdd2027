"""Stage-2 check: a 1.70 m avatar rendered at 3 walkable points per calibrated plate.
blender -b -P synth/scripts/check_calib.py -- render     -> synth/data/checks/calib/<plate>_<i>.png + points.json
python synth/scripts/check_calib.py score                -> synth/data/calib_check.csv (+ sheets of examples)
Score: the avatar is composited on the plate's middle frame and detected with YOLO11x; its box height is compared
with real people at the same foot point (robust linear fit h = a + b*v + c*u of the calibration's own inlier
boxes: data, not the camera model). Pass: |ratio - 1| <= 15% at all 3 points and the box bottom within
10% of the person's pixel height (at least 4 px) of the projected foot. Points: walkable-polygon interior points with the whole person visible,
nearest / median / farthest in depth."""
import sys, json, pathlib, os
import numpy as np

SYN = pathlib.Path(__file__).resolve().parents[1]
OUT = SYN / "data/checks/calib"
AVATAR, TARGET_H = "Male_Adult_05", 1.70
MIN_PX = 0.06  # check (and use) only spots where a 1.7 m person is >= 6% of the image height


def person_height_px(cam, P):
    foot, head, _ = cam.person_px(P, 1.7)
    return foot[:, 1] - head[:, 1]


def points(cal, cam, rng):
    from camera import inside
    poly = np.asarray(cal["walkable"], float)
    P = rng.uniform(poly.min(0), poly.max(0), (4000, 2))
    P = P[inside(poly, P) & cam.visible(P, 1.8, 0.03) & (person_height_px(cam, P) >= MIN_PX * cam.H)]
    if len(P) < 3:
        return None
    _, _, z = cam.person_px(P)
    o = np.argsort(z)
    return P[[o[0], o[len(o) // 2], o[-1]]]


def render():
    import bpy, mathutils
    sys.path.insert(0, str(SYN / "gen"))
    import rb, motion
    from camera import Camera
    OUT.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"; prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == "OPTIX"
    sc.cycles.device = "GPU"; sc.cycles.samples = 8; sc.cycles.use_denoising = True
    sc.render.film_transparent = True; sc.view_settings.view_transform = "Standard"
    world = bpy.data.worlds.new("w"); sc.world = world; world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[1].default_value = 1.5
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
    arm = rb.import_avatar(AVATAR)
    root = bpy.data.objects.new("root", None); sc.collection.objects.link(root); arm.parent = root
    h = 1.81  # Male_Adult_05 mesh height (synth/data/avatar_table.csv)
    import csv
    for r in csv.DictReader(open(SYN / "data/avatar_table.csv")):
        if r["avatar"] == AVATAR:
            h = float(r["height_m"])
    root.scale = (TARGET_H / h,) * 3
    motion.bake(arm, root, [dict(anim="m_idle_neutral_01", t0=-1, t1=1e9, mode="once")], [(0, 0, 0, 0)], "m", 30, 1, "chk")
    root.animation_data_clear()  # the bake keys the root; here it is placed by hand
    rng = np.random.default_rng(0)
    rec = {}
    for f in sorted((SYN / "calib").glob("*.json")):
        cal = json.loads(f.read_text())
        W, H = cal["render_size"]
        sc.render.resolution_x, sc.render.resolution_y = W, H
        cam.matrix_world = mathutils.Matrix(cal["camera_world"])
        cam.data.sensor_fit = "HORIZONTAL"; cam.data.sensor_width = 36.0
        cam.data.lens = cal["focal_px"] * W / cal["image_size"][0] * 36.0 / W
        C = Camera(cal, (W, H))
        P = points(cal, C, rng)
        if P is None:
            rec[cal["plate_id"]] = None; continue
        rec[cal["plate_id"]] = []
        for i, p in enumerate(P):
            # face the camera (sideways views make thin boxes)
            yaw = np.arctan2(C.C[1] - p[1], C.C[0] - p[0])
            root.location = (p[0], p[1], 0); root.rotation_euler = (0, 0, yaw + np.pi / 2)
            sc.frame_set(1)
            sc.render.filepath = str(OUT / f"{cal['plate_id']}_{i}.png")
            bpy.ops.render.render(write_still=True)
            foot = C.project(np.array([[p[0], p[1], 0.0]]))[0][0]
            rec[cal["plate_id"]].append(dict(xy=p.tolist(), foot=foot.tolist()))
    (OUT / "points.json").write_text(json.dumps(rec))


def score():
    os.environ.setdefault("YOLO_AUTOINSTALL", "false"); os.environ.setdefault("YOLO_OFFLINE", "true")
    import cv2, pandas as pd
    from ultralytics import YOLO
    sys.path.insert(0, str(SYN / "scripts"))
    from contact_sheet import frames_at, sheet
    model = YOLO(str(SYN / "assets/yolo11x.pt"))
    pts = json.loads((OUT / "points.json").read_text())
    plates = pd.read_csv(SYN / "data/plates.csv").set_index("plate_id")
    rows, ex_imgs, ex_caps = [], [], []
    for pid, P in pts.items():
        cal = json.loads((SYN / f"calib/{pid}.json").read_text())
        W, H = cal["render_size"]; k = W / cal["image_size"][0]
        fb = cal["fit_boxes"]
        u, vb, vt = (np.array(fb[x]) * k for x in ("u", "vb", "vt"))
        A = np.c_[np.ones_like(vb), vb, u]
        coef = np.linalg.lstsq(A, vb - vt, rcond=None)[0]
        for _ in range(3):  # robust refit
            res = (vb - vt) - A @ coef
            keep = np.abs(res) < 2.5 * 1.4826 * np.median(np.abs(res)) + 1e-6
            coef = np.linalg.lstsq(A[keep], (vb - vt)[keep], rcond=None)[0]
        if P is None:
            rows.append(dict(plate_id=pid, ok=False, why="no visible walkable points")); continue
        pr = plates.loc[pid]
        _, bg = frames_at(pr.path, [(pr.start + pr.end) / 2], int(max(pr.w, pr.h)))[0]
        bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_AREA)
        ratios, feet, why = [], [], ""
        for i, p in enumerate(P):
            im = cv2.imread(str(OUT / f"{pid}_{i}.png"), cv2.IMREAD_UNCHANGED)
            rgb, a = cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32), im[..., 3:4] / 255.0
            comp = (rgb * a + bg * (1 - a)).astype(np.uint8)
            fu, fv = p["foot"]
            r = model.predict(cv2.cvtColor(comp, cv2.COLOR_RGB2BGR), imgsz=1280, conf=0.25, classes=[0], verbose=False)[0]
            bx = r.boxes.xyxy.cpu().numpy()
            if not len(bx):
                why = "avatar not detected"; ratios.append(np.nan); feet.append(np.nan); continue
            j = np.argmin(np.abs((bx[:, 0] + bx[:, 2]) / 2 - fu) + np.abs(bx[:, 3] - fv))
            x0, y0, x1, y1 = bx[j]
            ref = coef @ [1, fv, fu]
            ratios.append((y1 - y0) / ref if ref > 4 else np.nan); feet.append(abs(y1 - fv) / max(ref, 40.0))
            if len(ex_imgs) < 48 and i == 1:
                c2 = comp.copy(); cv2.rectangle(c2, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 2)
                cv2.line(c2, (int(fu) - 15, int(fv)), (int(fu) + 15, int(fv)), (0, 255, 0), 2)
                cv2.line(c2, (int(fu), int(fv - ref)), (int(fu), int(fv)), (0, 255, 255), 1)
                ex_imgs.append(cv2.resize(c2, (480, int(480 * H / W)))); ex_caps.append(f"{pid[:28]} r={ratios[-1]:.2f}")
        ratios, feet = np.array(ratios), np.array(feet)
        ok_h = np.all(np.abs(ratios - 1) <= 0.15)
        ok_f = np.all(feet <= 0.10)  # box bottom within 10% of the person height (at least 4 px) of the foot point
        why = why or ("height off" if not ok_h else "feet off" if not ok_f else "")
        rows.append(dict(plate_id=pid, ok=not why, why=why, r_near=ratios[0], r_mid=ratios[1], r_far=ratios[2],
                         foot_err_rel_max=np.nanmax(feet) if np.isfinite(feet).any() else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(SYN / "data/calib_check.csv", index=False)
    for s in range(0, len(ex_imgs), 12):
        sheet(ex_imgs[s:s + 12], ex_caps[s:s + 12], OUT.parent / f"calib_examples_{s // 12}.jpg", cols=3)
    print(df.why.replace("", "ok").value_counts().to_string())
    print("pass", int(df.ok.sum()), "of", len(df), "| median |r-1|",
          float(np.nanmedian(np.abs(df[["r_near", "r_mid", "r_far"]].to_numpy() - 1))) if len(df) else None)


if __name__ == "__main__":
    sys.path.insert(0, str(SYN / "gen"))
    mode = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else sys.argv[1]
    render() if mode == "render" else score()
