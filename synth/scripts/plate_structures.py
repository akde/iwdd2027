"""Per-view structure masks (D29 fixes F4 and F6 part B): annotation protocol, reference images, label rasters, checks.

python synth/scripts/plate_structures.py refs VIEW [VIEW ...]     reference images to draw from (synth/data/structures/refs/)
python synth/scripts/plate_structures.py raster VIEW [VIEW ...]   label PNGs synth/data/structures/<plate_id>.png
python synth/scripts/plate_structures.py overlay VIEW [VIEW ...]  check sheet synth/data/structures/overlay_<view>.jpg
python synth/scripts/plate_structures.py check                    coverage, verdicts, post-event entries, hidden feet

STRUCTURE MASK PROTOCOL (per camera view). File synth/calib/structures/<view>.json:
{"view": "<view>", "plates": [<plate ids of the view>], "polygons": [{"layer": "S"|"F"|"N", "note": "...", "pts": [[x,y],...]}],
 "plate_adds": {"<plate_id>": [{"layer":..., "note":..., "pts": [...]}]}, "base": {"<polygon note>": [[x,y],...]},
 "verdict": "ok"|"trim"|"drop", "verdict_note": "...", "author": "<name>", "reviewed_by": null, "date": "2026-10-07"}
A view is the contest video id (its _p and _x plates share one file) or meva_<camera> for MEVA plates (the plate's
scene_id). Plates = the plates of plates.csv that pass synth/data/calib_check.csv; every one must be listed in exactly
one view file (check). Coordinates are normalised (x / render width, y / render height) on the plate's render-size
frame; the refs grid is in these units.
Layers:
- S = standing structure on the ground: building walls/faces, pillars, columns, poles wider than ~8 px, fences/railings
  that cannot be walked through, hedges/bushes, appliances (dryer), dumped piles and furniture, bins, benches,
  cabinets, post boxes, stairs' risers, gates. A walker whose feet are ABOVE the structure's ground-contact row in that
  column is behind it (forbidden). Draw the polygon so its BOTTOM edge follows the structure's real ground contact.
- F = front layer: a near-camera structure in front of everything (near roof apex/edge at the frame bottom,
  canopy/housing at the frame edge, foreground planter at the frame bottom): anything overlapping it is forbidden.
- N = no-go ground: ground-level areas where feet or objects must not be although they look walkable: dryer top,
  stairs/treads and raised landings, raised slabs and plinths, pile footprints, under a canopy whose fascia would be
  drawn behind a walker's head, glass-door tracks, water, flower beds. The black area of letterbox/pillarbox bars is
  handled elsewhere (F2 content box): do not draw bars.
- Overhangs (canopy fascia over walkable ground): label F only the part that is truly in front; give the canopy's real
  ground footprint (pillar bases) as N, or add a "base" polyline under the polygon's note that overrides its ground
  contact (contact row = max(vertical-run bottom, base row at that column); the polyline is extended flat beyond its
  ends). Notes used in "base" must name S polygons of the view or of the plate's plate_adds.
- Do NOT draw vehicles (separate vehicle masks exist). Thin things < ~8 px wide (single thin poles, wires, branches)
  are not drawn.
- plate_adds: things present only in some plates of the view, e.g. waste dumped in the SOURCE clip that is visible in
  plates whose window starts after the source clip's event (post-event plates, incl. extended _x plates), or a parked
  object that moved. Every post-event plate needs a plate_adds key (an empty list = checked, nothing to add); `refs`
  writes an absdiff proposal (pre-event vs plate median) for each of them.
- verdict: ok (masks cover everything relevant), trim (usable with these masks), drop (unusable: e.g. stairs/ramps
  everywhere, ground plane model broken, hardly any walkable ground left). Explain in verdict_note. author "prototype"
  = not verified, not used for sampling until it has been checked and completed.
- Gate (review F4): if a plate hides more than 20% of its real inlier feet (check, column feet_hidden), justify the
  mask in verdict_note (dump site, stairs) or redraw it.

Outputs (synth/data/structures/, derived, gitignored):
- refs/<plate_id>.png: 2x render size; middle frame of the plate window (MEVA: plus the middle frame of a plate of the
  other light, day/night, of the same view, side by side), grid every 0.05 (labels every 0.1, normalised), projected
  walkable polygon (yellow), real inlier feet of the calibration (cyan), still parked vehicles (translucent blue;
  outlined only: parked in fewer than all 9 frames), and the D29 evidence: for every pair of
  synth/data/exclude_pairs.csv on the plate, X (red circle + pair id + source V visual, L leg_overlap, B bar_check,
  C clone, P plate_view) and the walker tracks of both members (actor red, distractors magenta; postqc.walkers).
  refs/blind/<plate_id>.png: the same without the D29 evidence (for a blind first pass, review F4).
  refs/<plate_id>_postevent.png: post-event plates only: pre-event median (frames before the source clip's onset g)
  | plate-window median with the absdiff (red heat, contours) as a proposal for plate_adds.
- <plate_id>.png: uint8 labels at the plate's render size, 0 none, 1 S, 2 F, 3 N (N wins over S, F wins over all).
  <plate_id>_contact.png: uint16 ground contact for the depth rule, 0 = no structure, row + 1 for S pixels (vertical
  run bottom of the S layer, base overrides applied; N painted over S keeps the S contact), 65535 for F pixels.
  Depth rule (same as composite.occluded for still vehicles): a foot or object pixel at row v is behind/on a structure
  iff contact - 1 > v + 0.01 H, or it is on F or N. The rules and the rasteriser live in synth/gen/structures.py,
  which the generator uses (sampling, pre-render screen, compositor; DECISIONS.md D34).
- check.csv (check): plate, view, verdict, author, post-event flag and entry, raster state, share of real inlier feet
  and of the in-view walkable polygon (0.25 m grid) hidden by the masks."""
import json, pathlib, sys
import cv2, numpy as np, pandas as pd

SYN = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SYN / "gen"), str(SYN / "scripts")]
from camera import Camera, inside
import structures  # the one implementation of the rules, shared with the generator (gen/structures.py, D34)
from structures import LAYERS, FRONT, MARGIN, poly_mask, run_bottom, plate_polys, rasterize, hidden  # noqa: F401

ROOT = SYN.parent
SDIR = SYN / "calib/structures"
ODIR = SYN / "data/structures"
SRC = {"visual": "V", "leg_overlap": "L", "bar_check": "B", "clone": "C", "plate_view": "P"}
COL = {1: (255, 140, 0), 2: (0, 210, 0), 3: (0, 110, 255)}  # RGB: S orange, F green, N blue


# ---------------------------------------------------------------- tables
def plate_table():
    """Plates of plates.csv that pass calib_check.csv and have a calib JSON, with their view and render size."""
    pl = pd.read_csv(SYN / "data/plates.csv", dtype={"label": str})
    chk = pd.read_csv(SYN / "data/calib_check.csv")
    ok = set(chk[chk.ok].plate_id)
    pl = pl[[p in ok and (SYN / f"calib/{p}.json").exists() for p in pl.plate_id]].copy()
    pl["view"] = np.where(pl.source == "meva", pl.scene_id, pl.video)
    rs = [json.loads((SYN / f"calib/{p}.json").read_text())["render_size"] for p in pl.plate_id]
    pl["W"], pl["H"] = [r[0] for r in rs], [r[1] for r in rs]
    return pl.set_index("plate_id")


def calib(pid):
    return json.loads((SYN / f"calib/{pid}.json").read_text())


def load_view(view):
    f = SDIR / f"{view}.json"
    return json.loads(f.read_text()) if f.exists() else None


def view_plates(view, tab):
    """The view JSON's plates (those still passing calib_check; check reports the others), else the derived list."""
    js = load_view(view)
    return [p for p in js["plates"] if p in tab.index] if js else sorted(tab[tab.view == view].index)


def post_event(r):
    """(is post-event, g): the plate window starts after its source clip's onset (x-plates included)."""
    return structures.post_event(r.source, r.video, r.start)


def night(pid):
    """MEVA clip hour from the id (2018-03-11_23-55-...): night if >= 19 h or < 6 h."""
    h = int(pid[11:13])
    return h >= 19 or h < 6


# ---------------------------------------------------------------- drawing helpers
def frames(r, ts, W, H):
    from contact_sheet import frames_at
    return [cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA) for _, f in frames_at(r.path, ts, max(r.w, r.h))]


def text(img, s, xy, col=(255, 255, 255), scale=0.5):
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


def grid(img):
    """Normalised grid: lines every 0.05 (labels every 0.1) on all four sides."""
    H, W = img.shape[:2]
    ov = img.copy()
    for k in range(1, 20):
        x, y = round(k * 0.05 * W), round(k * 0.05 * H)
        c = (255, 255, 255) if k % 2 == 0 else (170, 170, 170)
        cv2.line(ov, (x, 0), (x, H), c, 1)
        cv2.line(ov, (0, y), (W, y), c, 1)
    img[:] = cv2.addWeighted(ov, 0.55, img, 0.45, 0)
    for k in range(2, 20, 2):
        x, y, s = round(k * 0.05 * W), round(k * 0.05 * H), f"{k * 0.05:.1f}"
        for xy in ((x + 2, 14), (x + 2, H - 4)):
            text(img, s, xy, (255, 255, 0), 0.45)
        for xy in ((2, y - 3), (W - 30, y - 3)):
            text(img, s, xy, (255, 255, 0), 0.45)
    return img


def ground_line(img, cam, xy, s, col, closed=False, thick=1):
    uv, z = cam.project(np.c_[xy, np.zeros(len(xy))])
    ok = (z > 0.3) & (np.abs(uv) < 1e5).all(1)
    q = np.round(uv * s).astype(int)
    for i in range(len(xy) if closed else len(xy) - 1):
        j = (i + 1) % len(xy)
        if ok[i] and ok[j]:
            cv2.line(img, tuple(q[i]), tuple(q[j]), col, thick, cv2.LINE_AA)


def walkable_line(img, cal, cam, s):
    from scenario import boundary_points
    ground_line(img, cam, boundary_points(np.asarray(cal["walkable"]), 0.1), s, (255, 230, 0), True, 2)


def feet_px(cal):
    fb = cal["fit_boxes"]; k = cal["render_size"][0] / cal["image_size"][0]
    return np.c_[fb["u"], fb["vb"]] * k


def vehicles(img, pid):
    f = SYN / f"data/vehicles/{pid}.png"
    if not f.exists():
        return
    v = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
    v = cv2.resize(v, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    meta = {d["id"]: d for d in json.loads((SYN / "data/vehicles.json").read_text()).get(pid, [])}
    ov = img.copy()
    for k in np.unique(v[v > 0]):
        d = meta.get(int(k), {})
        mk = (v == k).astype(np.uint8)
        if d.get("n_frames", 0) == d.get("n_of", -1):
            ov[mk > 0] = (40, 90, 255)
        cs, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, (40, 90, 255), 2)
        cv2.drawContours(ov, cs, -1, (40, 90, 255), 2)
    img[:] = cv2.addWeighted(ov, 0.4, img, 0.6, 0)


def d29_evidence():
    """plate_id -> [(pair, source letters, group, [members])] for the pairs of exclude_pairs.csv."""
    ex = pd.read_csv(SYN / "data/exclude_pairs.csv")
    out = {}
    for r in ex.itertuples():
        sd = SYN / ("data/batch_s34/scenarios" if r.batch == "batch_s34" else "data/scenarios")
        S = [json.loads(f.read_text()) for f in sorted(sd.glob(f"{r.pair_id}_*.json"))]
        if not S:
            print(f"warning: no scenario JSON for {r.pair_id} in {sd}")
            continue
        src = "".join(SRC.get(x, "?") for x in r.source.split(";"))
        gr = S[0]["group"]
        out.setdefault(gr["plate_id"], []).append((r.pair_id, src, gr, [s["member"] for s in S]))
    return out


def draw_d29(img, cal, items, s):
    """D29 evidence with the camera each group was sampled with (camera.group_calib), scaled to this panel."""
    from camera import group_calib
    from tracks import walkers
    for pair, src, gr, ms in items:
        cam, k = Camera(group_calib(gr), gr["render_size"]), s * cal["render_size"][0] / gr["render_size"][0]
        for m in ms:
            for i, w in enumerate(walkers(gr, m)):
                ground_line(img, cam, w, k, (255, 0, 0) if i == 0 else (255, 0, 255), False, 2 if i == 0 else 1)
    for pair, src, gr, ms in items:
        cam, k = Camera(group_calib(gr), gr["render_size"]), s * cal["render_size"][0] / gr["render_size"][0]
        (u, v), = cam.project(np.array([[*gr["path"]["X"], 0.0]]))[0]
        q = (int(u * k), int(v * k))
        cv2.circle(img, q, 9, (255, 0, 0), 2, cv2.LINE_AA)
        text(img, f"{pair} {src}", (q[0] + 10, q[1] - 4), (255, 80, 80), 0.5)


def strip(width, lines):
    import textwrap
    lines = [w for l in lines for w in textwrap.wrap(l, max(20, width // 10))]
    out = np.zeros((22 * len(lines) + 6, width, 3), np.uint8)
    for i, l in enumerate(lines):
        text(out, l, (6, 20 + 22 * i), (255, 255, 0), 0.55)
    return out


def hstack(ims, gap=8):
    h = max(i.shape[0] for i in ims)
    ims = [np.pad(i, ((0, h - i.shape[0]), (0, 0), (0, 0))) for i in ims]
    sep = np.full((h, gap, 3), 255, np.uint8)
    return np.hstack(sum([[i, sep] for i in ims], [])[:-1])


def vstack(ims):
    w = max(i.shape[1] for i in ims)
    return np.vstack([np.pad(i, ((0, 0), (0, w - i.shape[1]), (0, 0))) for i in ims])


def write(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".png":
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 6])
    else:
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])


# ---------------------------------------------------------------- commands
def cmd_refs(view, tab, ev):
    pids = view_plates(view, tab)
    if not pids:
        sys.exit(f"no plates for view {view}")
    S, mids = 2, {}

    def mid(pid):
        if pid not in mids:
            r = tab.loc[pid]
            mids[pid] = frames(r, [(r.start + r.end) / 2], r.W, r.H)[0]
        return mids[pid]

    def panel(pid):
        r, cal = tab.loc[pid], calib(pid)
        cam = Camera(cal, cal["render_size"])
        img = cv2.resize(mid(pid), (r.W * S, r.H * S), interpolation=cv2.INTER_CUBIC)
        vehicles(img, pid)
        walkable_line(img, cal, cam, S)
        for u, v in feet_px(cal):
            cv2.circle(img, (int(u * S), int(v * S)), 3, (0, 255, 255), -1, cv2.LINE_AA)
        return img, cal

    print(f"{view}: {len(pids)} plates")
    for pid in pids:
        r = tab.loc[pid]
        pe, g = post_event(r)
        img, cal = panel(pid)
        blind = grid(img.copy())
        draw_d29(img, cal, ev.get(pid, []), S)
        main = grid(img)
        light = ("night" if night(pid) else "day") if r.source == "meva" else ""
        comp = None
        if r.source == "meva":
            other = [p for p in pids if night(p) != night(pid)]
            if other:
                comp = other[len(other) // 2]
                cimg = grid(panel(comp)[0])
        n_ev = len(ev.get(pid, []))
        title = (f"{pid}  view {view}  render {r.W}x{r.H}  window {r.start:.1f}-{r.end:.1f} s  {light}"
                 + (f"  POST-EVENT (onset g={g:.0f} s)" if pe else ""))
        legend = ("yellow: walkable polygon  cyan: real feet  blue: still parked vehicles (outline only: not still)"
                  + (f"  right: {'night' if night(comp) else 'day'} plate {comp}" if comp else ""))
        for im, out, hd in ((main, ODIR / f"refs/{pid}.png",
                             [title + f"  D29 pairs: {n_ev}", legend, "red: D29 actor tracks, X + pair id (source V "
                              "visual, L leg overlap, B black bar, C clone, P plate view)  magenta: D29 distractors"]),
                            (blind, ODIR / f"refs/blind/{pid}.png", [title + "  (blind: no D29 evidence)", legend])):
            body = hstack([im, cimg]) if comp else im
            write(out, vstack([strip(body.shape[1], hd), body]))
        if pe:
            post_event_ref(r, pid, g, S)
        print(f"  {pid}  {r.W}x{r.H}  {r.start:.1f}-{r.end:.1f} s  {light}  D29 pairs {n_ev}"
              + (f"  POST-EVENT (g={g:.0f} s)" if pe else ""))


def post_event_ref(r, pid, g, S):
    """Pre-event median (source clip before g) | plate-window median with the absdiff as a plate_adds proposal."""
    t1 = max(g - 0.5, 0.2)
    pre = np.median(np.stack(frames(r, list(np.linspace(min(0.25, t1), t1, 9)), r.W, r.H)), 0).astype(np.uint8)
    post = np.median(np.stack(frames(r, list(np.linspace(r.start + 0.1, r.end - 0.1, 9)), r.W, r.H)), 0).astype(np.uint8)
    z = [cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), (5, 5), 0).astype(float) for a in (pre, post)]
    d = np.abs(np.subtract(*[(a - a.mean()) / max(a.std(), 1.0) for a in z]))  # grey levels in std units: no exposure
    big = lambda a: cv2.resize(a, (r.W * S, r.H * S), interpolation=cv2.INTER_CUBIC)
    heat = cv2.resize(np.clip(d, 0, 1), (r.W * S, r.H * S))[..., None]
    ov = (big(post) * (1 - 0.6 * heat) + np.array([255, 0, 0]) * 0.6 * heat).astype(np.uint8)
    m = cv2.morphologyEx((d > 0.6).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cs, _ = cv2.findContours(cv2.resize(m, (r.W * S, r.H * S), interpolation=cv2.INTER_NEAREST),
                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = [c for c in cs if cv2.contourArea(c) > 0.002 * r.W * r.H * S * S]
    cv2.drawContours(ov, cs, -1, (255, 255, 0), 2)
    body = hstack([grid(big(pre)), grid(ov)])
    hd = [f"{pid}  POST-EVENT: left = median of 9 frames {min(0.25, t1):.1f}-{t1:.1f} s (before onset g={g:.0f} s), "
          f"right = median of the plate window {r.start:.1f}-{r.end:.1f} s",
          "red heat = |difference| of the grey images, each normalised to mean 0 / std 1 (full at 1 std), yellow "
          "contours > 0.6 std: candidates for plate_adds (dumped waste, moved objects; the dumper's vehicle leaving "
          "and local lighting also show)"]
    write(ODIR / f"refs/{pid}_postevent.png", vstack([strip(body.shape[1], hd), body]))


def cmd_raster(view, tab):
    js = load_view(view)
    if js is None:
        sys.exit(f"no {SDIR / f'{view}.json'}")
    ODIR.mkdir(parents=True, exist_ok=True)
    for pid in js["plates"]:
        W, H = calib(pid)["render_size"]
        lab, contact = rasterize(js, pid, W, H)
        cv2.imwrite(str(ODIR / f"{pid}.png"), lab)
        cv2.imwrite(str(ODIR / f"{pid}_contact.png"), contact)
        sh = {k: float((lab == v).mean()) for k, v in LAYERS.items()}
        print(f"{pid}  {W}x{H}  S {sh['S']:.1%}  F {sh['F']:.1%}  N {sh['N']:.1%}"
              + (f"  (+{len(js['plate_adds'][pid])} plate_adds)" if pid in (js.get("plate_adds") or {}) else ""))


def hidden_shares(js, pid, cal):
    """Share of the real inlier feet and of the in-view walkable polygon (0.25 m grid) hidden by the masks."""
    W, H = cal["render_size"]
    lab, contact = rasterize(js, pid, W, H)
    cam = Camera(cal, (W, H))
    fh = hidden(lab, contact, feet_px(cal)).mean()
    poly = np.asarray(cal["walkable"])
    xs, ys = np.meshgrid(np.arange(poly[:, 0].min(), poly[:, 0].max(), 0.25),
                         np.arange(poly[:, 1].min(), poly[:, 1].max(), 0.25))
    g = np.c_[xs.ravel(), ys.ravel()]
    g = g[inside(poly, g)]
    uv, z = cam.project(np.c_[g, np.zeros(len(g))])
    ok = (z > 0.3) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    wh = hidden(lab, contact, uv[ok]).mean() if ok.any() else float("nan")
    return float(fh), float(wh)


def cmd_overlay(view, tab, ev):
    js = load_view(view)
    if js is None:
        sys.exit(f"no {SDIR / f'{view}.json'}")
    tiles = []
    for pid in view_plates(view, tab):
        r, cal = tab.loc[pid], calib(pid)
        W, H = cal["render_size"]; s = 2 if W < 640 else 1
        lab, contact = rasterize(js, pid, W, H)
        img = cv2.resize(frames(r, [(r.start + r.end) / 2], W, H)[0], (W * s, H * s), interpolation=cv2.INTER_CUBIC)
        L = cv2.resize(lab, (W * s, H * s), interpolation=cv2.INTER_NEAREST)
        ov = img.copy()
        for k, c in COL.items():
            ov[L == k] = c
        img = cv2.addWeighted(ov, 0.45, img, 0.55, 0)
        for k, c in COL.items():
            cs, _ = cv2.findContours((L == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cs, -1, c, 1)
        for note, line in (js.get("base") or {}).items():
            if any(p.get("note") == note for p in plate_polys(js, pid)):
                cv2.polylines(img, [np.round(np.asarray(line) * [W * s, H * s]).astype(np.int32)], False,
                              (255, 255, 255), 2)
        cam = Camera(cal, (W, H))
        walkable_line(img, cal, cam, s)
        f = feet_px(cal); hid = hidden(lab, contact, f)
        for (u, v), h in zip(f, hid):
            cv2.circle(img, (int(u * s), int(v * s)), 3 if h else 2, (255, 255, 255) if h else (0, 255, 255), -1)
        draw_d29(img, cal, ev.get(pid, []), s)
        fh, wh = hidden_shares(js, pid, cal)
        pe, _ = post_event(r)
        tiles.append((img, f"{pid[-34:]}{'  POST-EVENT' if pe else ''}  feet hidden {fh:.0%}  walkable hidden {wh:.0%}"))
    cols = 3 if tiles[0][0].shape[1] <= 700 else 2
    rows = []
    for i in range(0, len(tiles), cols):
        rows.append(hstack([vstack([strip(t.shape[1], [c]), t]) for t, c in tiles[i:i + cols]]))
    body = vstack(sum([[rw, np.full((8, rw.shape[1], 3), 255, np.uint8)] for rw in rows], [])[:-1])
    hd = [f"{view}: verdict {js.get('verdict')}  author {js.get('author')}  reviewed_by {js.get('reviewed_by')}",
          "orange S standing structure  green F front layer  blue N no-go ground  white line: base override  "
          "yellow: walkable  cyan: real feet (white: hidden by the masks)  red/magenta: D29 tracks"]
    out = ODIR / f"overlay_{view}.jpg"
    write(out, vstack([strip(body.shape[1], hd), body]))
    print(out)


def cmd_check(tab):
    views = {f.stem: json.loads(f.read_text()) for f in sorted(SDIR.glob("*.json"))}
    owner, rows, probs = {}, [], []
    for v, js in views.items():
        if js.get("view") != v:
            probs.append(f"{v}.json: 'view' is {js.get('view')!r}")
        if js.get("verdict") not in ("ok", "trim", "drop"):
            probs.append(f"{v}.json: verdict {js.get('verdict')!r}")
        for p in js.get("plates", []):
            owner.setdefault(p, []).append(v)
            if p not in tab.index:
                probs.append(f"{v}.json lists {p}, which is not a plate passing calib_check")
        for p in js.get("plate_adds") or {}:
            if p not in js.get("plates", []):
                probs.append(f"{v}.json: plate_adds for {p}, which is not in its plates")
    for v, n in tab[[p not in owner for p in tab.index]].groupby("view").size().items():
        probs.append(f"{v}: {n} plates in no view JSON" + ("" if v in views else f" (no {v}.json)"))
    for pid, r in tab.iterrows():
        vs = owner.get(pid, [])
        pe, g = post_event(r)
        row = dict(plate_id=pid, view=r.view, json=";".join(vs), post_event=pe, g=g)
        if len(vs) > 1:
            probs.append(f"{pid}: in {len(vs)} view JSONs {vs}")
        if len(vs) == 1:
            js = views[vs[0]]
            row.update(verdict=js.get("verdict"), author=js.get("author"), reviewed_by=js.get("reviewed_by"),
                       post_entry=pid in (js.get("plate_adds") or {}))
            if pe and not row["post_entry"] and js.get("verdict") != "drop":
                probs.append(f"{pid}: post-event plate (g={g:.0f} s, window from {r.start:.1f} s) without a "
                             f"plate_adds entry in {vs[0]}.json")
            f = ODIR / f"{pid}.png"
            jm = (SDIR / f"{vs[0]}.json").stat().st_mtime
            row["raster"] = "missing" if not f.exists() else ("stale" if f.stat().st_mtime < jm else "ok")
            try:
                row["feet_hidden"], row["walkable_hidden"] = hidden_shares(js, pid, calib(pid))
            except ValueError as e:
                probs.append(f"{vs[0]}.json ({pid}): {e}")
        rows.append(row)
    df = pd.DataFrame(rows)
    ODIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(ODIR / "check.csv", index=False)
    allv = sorted(set(tab.view))
    print(f"{len(tab)} plates in {len(allv)} views; view JSONs: {len(views)}")
    print("views without a JSON:", " ".join(v for v in allv if v not in views) or "none")
    print("views with verdict drop:", " ".join(v for v, js in views.items() if js.get("verdict") == "drop") or "none")
    print("views by author 'prototype' (not verified):",
          " ".join(v for v, js in views.items() if js.get("author") == "prototype") or "none")
    print("post-event plates:", " ".join(df[df.post_event].plate_id))
    if "feet_hidden" in df:
        hi = df[(df.feet_hidden > 0.2) & (df.verdict != "drop")]
        for r in hi.itertuples():
            print(f"gate: {r.plate_id} hides {r.feet_hidden:.0%} of its real feet: justify in verdict_note or redraw")
    if "raster" in df:
        bad = df[df.raster.isin(["missing", "stale"])]
        if len(bad):
            print("rasters missing or stale:", " ".join(f"{p}({s})" for p, s in zip(bad.plate_id, bad.raster)))
    for p in probs:
        print("PROBLEM", p)
    print(f"{len(probs)} problems; table {ODIR / 'check.csv'}")
    return 1 if probs else 0


def main(argv):
    if not argv or argv[0] not in ("refs", "raster", "overlay", "check"):
        sys.exit(__doc__.split("\n\n")[0])
    cmd, args = argv[0], argv[1:]
    tab = plate_table()
    if cmd == "check":
        return cmd_check(tab)
    if not args:
        sys.exit(f"{cmd} needs one or more views")
    ev = d29_evidence() if cmd in ("refs", "overlay") else {}
    for v in args:
        {"refs": lambda: cmd_refs(v, tab, ev), "raster": lambda: cmd_raster(v, tab),
         "overlay": lambda: cmd_overlay(v, tab, ev)}[cmd]()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
