"""Motion baking, run inside Blender.
Rocketbox animation files have a different rest pose from the avatars' bind pose, so local bone channels do not
transfer. Retargeting copies each bone's armature-space orientation from the animation armature to the avatar
(both are 3ds Max Biped rigs with the same bone axes) and keeps the avatar's own bone offsets. Root motion is on the
Bip01 object channels; it is made in-place (linear drift removed) and scaled by the avatar's hip height, and the
actor root empty carries the path.

pose_table(arm, anim, gender) -> PoseTable (native 30 fps frames)
bake(arm, root, segs, keys, fps, n) -> writes one action on the avatar armature and one on the root empty
Segments: dicts anim, t0, t1 (clip seconds), mode "loop" | "once", offset (native frames into the animation),
optional fit (play the whole animation in `fit` s) or rate.
An animation recorded on the other gender's skeleton (female avatars bend with the male crouch, DECISIONS.md D35) is
scaled by the root height it was recorded at (its name prefix m_ / f_), so the avatar's hips stay at its own height.
Root keys: list of (t, x, y, yaw) with yaw = facing angle in radians (0 = +X); linear in between."""
import math
import bpy, mathutils
import numpy as np
import rb

NATIVE_FPS = 30.0
STAND_Z = {"m": 0.895, "f": 0.919}  # Bip01 root height of the standing animations (synth/data/anim_inventory.csv)
_cache = {}


def _order(arm):
    out = []

    def walk(b):
        out.append(b.name)
        for c in b.children:
            walk(c)
    for b in arm.data.bones:
        if b.parent is None:
            walk(b)
    return out


def _anim_armature(name):
    before = set(bpy.data.objects); before_a = set(bpy.data.actions)
    bpy.ops.import_scene.fbx(filepath=str(rb.anim_path(name)), automatic_bone_orientation=False)
    new = [o for o in bpy.data.objects if o not in before]
    a = next(o for o in new if o.type == "ARMATURE")
    for o in new:
        if o is not a:
            bpy.data.objects.remove(o)
    keep = a.animation_data.action
    for x in [x for x in bpy.data.actions if x not in before_a and x is not keep]:
        bpy.data.actions.remove(x)
    return a, keep


class PoseTable:
    def __init__(self, bones, root_loc, root_rot, quat, pelvis_loc, speed, cyclic):
        self.bones, self.root_loc, self.root_rot, self.quat, self.pelvis_loc = bones, root_loc, root_rot, quat, pelvis_loc
        self.speed, self.cyclic, self.F = speed, cyclic, len(root_loc)

    def sample(self, fr, loop):
        """Pose at fractional native frame index fr (0-based): (root_loc, root_rot, quat [B,4], pelvis_loc)."""
        F = self.F
        if loop:
            fr = fr % (F - 1)  # first and last frames of a cycle are the same pose
        fr = min(max(fr, 0.0), F - 1.0)
        i = int(math.floor(fr)); j = min(i + 1, F - 1); w = fr - i
        q = _nlerp(self.quat[i], self.quat[j], w)
        return ((1 - w) * self.root_loc[i] + w * self.root_loc[j], (1 - w) * self.root_rot[i] + w * self.root_rot[j], q,
                (1 - w) * self.pelvis_loc[i] + w * self.pelvis_loc[j])


def _nlerp(a, b, w):
    b = np.where((np.sum(a * b, -1) < 0)[..., None], -b, b)
    q = (1 - w) * a + w * b
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def anim_gender(anim, gender):
    """The skeleton an animation was recorded on (name prefix m_ / f_); else the avatar's gender."""
    return anim[0] if anim[:2] in ("m_", "f_") else gender


def pose_table(arm, anim, gender):
    # per avatar (rb.import_avatar tags it): leg_overlap rebuilds many members in one process, all named "actor"
    key = (arm.name, arm.get("avatar"), gender, anim)
    if key in _cache:
        return _cache[key]
    src, act = _anim_armature(anim)
    sc = bpy.context.scene
    f0, f1 = (int(round(x)) for x in act.frame_range)
    order = _order(arm)
    rest = {b: arm.data.bones[b].matrix_local.copy() for b in order}
    par = {b: (arm.data.bones[b].parent.name if arm.data.bones[b].parent else None) for b in order}
    rel = {b: (rest[par[b]].inverted() @ rest[b]) if par[b] else None for b in order}
    s = (arm.location.z if arm.get("rest_z") is None else arm["rest_z"]) / STAND_Z[anim_gender(anim, gender)]
    R_loc, R_rot, Q, P = [], [], [], []
    for f in range(f0, f1 + 1):
        sc.frame_set(f)
        R_loc.append(np.array(src.location)); R_rot.append(np.array(src.rotation_euler))
        pose, qs = {}, []
        for b in order:
            sb = src.pose.bones.get(b)
            if sb is None:
                rot = (rest[b] if par[b] is None else pose[par[b]] @ rel[b]).to_quaternion()
            else:
                rot = sb.matrix.to_quaternion()
            if par[b] is None:
                head = sb.matrix.translation if sb is not None else rest[b].translation
                pose[b] = mathutils.Matrix.Translation(head) @ rot.to_matrix().to_4x4()
                basis = rest[b].inverted() @ pose[b]
                P.append(np.array(basis.translation) * s)
            else:
                head = (pose[par[b]] @ rel[b]).translation
                pose[b] = mathutils.Matrix.Translation(head) @ rot.to_matrix().to_4x4()
                basis = rel[b].inverted() @ pose[par[b]].inverted() @ pose[b]
            q = basis.to_quaternion(); qs.append([q.w, q.x, q.y, q.z])
        Q.append(qs)
    R_loc, R_rot, Q, P = np.array(R_loc), np.array(R_rot), np.array(Q), np.array(P)
    drift = R_loc[-1, :2] - R_loc[0, :2]
    lin = np.linspace(0, 1, len(R_loc))[:, None]
    cyclic = "_walk" in anim
    if cyclic:  # locomotion: in place, the path moves the root empty
        R_loc[:, :2] -= R_loc[0, :2] + lin * drift
    elif "crouch_out" in anim:  # body motion over planted feet: anchored where it ends (crouch_in leaves it there)
        R_loc[:, :2] -= R_loc[-1, :2]
    else:  # crouch_in, idle: anchored where it starts
        R_loc[:, :2] -= R_loc[0, :2]
    R_loc *= s  # hip-height scale (z: hip height, xy: residual sway)
    speed = float(np.linalg.norm(drift) * s / ((f1 - f0) / NATIVE_FPS))
    act_name = act.name
    bpy.data.objects.remove(src)
    if act_name in bpy.data.actions:
        bpy.data.actions.remove(bpy.data.actions[act_name])
    pt = PoseTable(order, R_loc, R_rot, Q, P, speed, cyclic=cyclic)
    _cache[key] = pt
    return pt


def _smooth(x):
    return x * x * (3 - 2 * x)


def evaluate(arm, segs, gender, t, xf=0.3):
    """Pose at clip time t: the segment containing t, cross-faded over xf s from the pose actually shown at that
    time by the previous segments (recursive, so an unfinished fade is never dropped)."""
    i = max([k for k, s in enumerate(segs) if s["t0"] <= t] or [0])

    def at(k):
        s = segs[k]
        pt = pose_table(arm, s["anim"], gender)
        rate = (pt.F - 1) / NATIVE_FPS / s["fit"] if "fit" in s else s.get("rate", 1.0)  # fit: play in `fit` s
        fr = s.get("offset", 0.0) + (t - s["t0"]) * NATIVE_FPS * rate
        return pt.sample(fr, s["mode"] == "loop")

    def shown(k):
        p = at(k)
        if k > 0 and t - segs[k]["t0"] < xf:
            w = _smooth((t - segs[k]["t0"]) / xf)
            q = shown(k - 1)
            p = tuple((1 - w) * a + w * b for a, b in zip(q[:2], p[:2])) + (_nlerp(q[2], p[2], w), (1 - w) * q[3] + w * p[3])
        return p
    return shown(i)


def _write(action, path, index, frames, values, group=None):
    fc = action.fcurves.new(path, index=index, action_group=group or "")
    kp = fc.keyframe_points
    kp.add(len(frames))
    kp.foreach_set("co", np.column_stack([frames, values]).astype(np.float32).ravel())
    kp.foreach_set("interpolation", [1] * len(frames))  # LINEAR
    fc.update()


def root_track(keys, times):
    """Piecewise-linear x, y and yaw (unwrapped) at the given times."""
    k = np.array(keys, float)
    yaw = np.unwrap(k[:, 3])
    return np.interp(times, k[:, 0], k[:, 1]), np.interp(times, k[:, 0], k[:, 2]), np.interp(times, k[:, 0], yaw)


def bake(arm, root, segs, keys, gender, fps, n, name):
    """Frames 1..n at times (frame - 1) / fps. The avatar faces -Y in its parent frame (anim root yaw -90 deg),
    so the root empty's z rotation is the facing angle + 90 deg."""
    times = np.arange(n) / fps
    frames = np.arange(1, n + 1, dtype=float)
    poses = [evaluate(arm, segs, gender, t) for t in times]
    bones = pose_table(arm, segs[0]["anim"], gender).bones
    act = bpy.data.actions.new(name + "_arm")
    for c in range(3):
        _write(act, "location", c, frames, [p[0][c] for p in poses], "root")
        _write(act, "rotation_euler", c, frames, [p[1][c] for p in poses], "root")
    for bi, b in enumerate(bones):
        for c in range(4):
            _write(act, f'pose.bones["{b}"].rotation_quaternion', c, frames, [p[2][bi][c] for p in poses], b)
        if bi == 0:
            for c in range(3):
                _write(act, f'pose.bones["{b}"].location', c, frames, [p[3][c] for p in poses], b)
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
    arm.animation_data_create(); arm.animation_data.action = act
    x, y, yaw = root_track(keys, times)
    ract = bpy.data.actions.new(name + "_root")
    _write(ract, "location", 0, frames, x); _write(ract, "location", 1, frames, y)
    _write(ract, "location", 2, frames, np.zeros(n))
    _write(ract, "rotation_euler", 2, frames, yaw + math.pi / 2)
    root.animation_data_create(); root.animation_data.action = ract
    return times
