"""Rocketbox helpers, run inside Blender (bpy). Avatars and animations share the 3ds Max Biped skeleton (Bip01 ...),
so an action imported from any animation FBX drives any avatar armature by bone name.
import_avatar(name)      -> armature object; textures relinked from the avatar's Textures/ folder, opacity via alpha.
import_action(name)      -> bpy action (bone channels only), named after the animation.
in_place(arm, action)    -> (action without the pelvis' horizontal drift, drift vector in m, duration in s)."""
import pathlib
import bpy, mathutils

RB = pathlib.Path(__file__).resolve().parents[1] / "assets/rocketbox/Assets"
ANIM_DIRS = ["all_animations_max_motextr_xy", "all_animations_max_motextr_static", "all_animations_max_motextr_xyz"]


def avatar_dir(name):
    for sub in ("Adults", "Professions", "Children"):
        p = RB / "Avatars" / sub / name
        if p.exists():
            return p
    raise FileNotFoundError(name)


def _fix_materials(meshes, tex_dir):
    for o in meshes:
        for m in o.data.materials:
            nt = m.node_tree
            bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
            for n in [n for n in nt.nodes if n.type == "TEX_IMAGE"]:
                if n.image is None:
                    continue
                f = tex_dir / pathlib.Path(n.image.filepath).name
                if not f.exists():
                    continue
                n.image = bpy.data.images.load(str(f), check_existing=True)
                if "_normal" in f.name or "_specular" in f.name:
                    n.image.colorspace_settings.name = "Non-Color"
                if "_specular" in f.name:  # specular maps read as gloss in Unity; keep the surface matte
                    for l in list(n.outputs[0].links):
                        nt.links.remove(l)
                if "_opacity" in f.name or "opacity" in m.name:
                    nt.links.new(n.outputs["Alpha"], bsdf.inputs["Alpha"])
            bsdf.inputs["Roughness"].default_value = 0.75
            if "Specular IOR Level" in bsdf.inputs:
                bsdf.inputs["Specular IOR Level"].default_value = 0.3
            m.blend_method = "HASHED"


def import_avatar(name):
    before = set(bpy.data.objects)
    fbx = avatar_dir(name) / "Export" / f"{name}.fbx"
    bpy.ops.import_scene.fbx(filepath=str(fbx), automatic_bone_orientation=False)
    new = [o for o in bpy.data.objects if o not in before]
    arm = next(o for o in new if o.type == "ARMATURE")
    for o in new:  # Footsteps helper etc.
        if o.type == "EMPTY":
            bpy.data.objects.remove(o)
    meshes = [o for o in arm.children if o.type == "MESH"]
    _fix_materials(meshes, avatar_dir(name) / "Textures")
    arm.name = name
    arm["avatar"] = name  # survives renaming (build_scene.make_actor): motion's pose-table cache is per avatar
    if arm.animation_data:
        arm.animation_data.action = None
    return arm


def anim_path(name):
    for d in ANIM_DIRS:
        p = RB / "Animations" / d / f"{name}.max.fbx"
        if p.exists():
            return p
    raise FileNotFoundError(name)


def import_action(name):
    if name in bpy.data.actions:
        return bpy.data.actions[name]
    before_o, before_a = set(bpy.data.objects), set(bpy.data.actions)
    bpy.ops.import_scene.fbx(filepath=str(anim_path(name)), automatic_bone_orientation=False)
    new_o = [o for o in bpy.data.objects if o not in before_o]
    arm = next(o for o in new_o if o.type == "ARMATURE")
    act = arm.animation_data.action
    # object-level channels (location, rotation_euler, scale of the Bip01 root) carry the root motion
    act.name = name
    act.use_fake_user = True
    for o in new_o:
        bpy.data.objects.remove(o)
    for a in [a for a in bpy.data.actions if a not in before_a and a is not act]:
        bpy.data.actions.remove(a)
    return act


def pelvis_world(arm, action, frame):
    arm.animation_data_create(); arm.animation_data.action = action
    bpy.context.scene.frame_set(int(frame), subframe=frame - int(frame))
    return arm.matrix_world @ arm.pose.bones["Bip01 Pelvis"].head


def in_place(arm, action, keep_z=True):
    """Bake a copy of `action` whose pelvis has no net horizontal drift (linear drift removed per frame).
    Returns (new action, drift vector start->end in world m, duration s at the scene fps)."""
    f0, f1 = (int(round(x)) for x in action.frame_range)
    sc = bpy.context.scene
    p0, p1 = pelvis_world(arm, action, f0), pelvis_world(arm, action, f1)
    drift = p1 - p0; drift.z = 0.0
    new = action.copy(); new.name = action.name + "_inplace"; new.use_fake_user = True
    pb = arm.pose.bones["Bip01 Pelvis"]
    targets = {}
    for f in range(f0, f1 + 1):
        arm.animation_data.action = action
        sc.frame_set(f)
        w = arm.matrix_world @ pb.matrix
        w.translation -= drift * ((f - f0) / max(1, f1 - f0))
        targets[f] = arm.matrix_world.inverted() @ w
    for c in [c for c in new.fcurves if c.data_path == 'pose.bones["Bip01 Pelvis"].location']:
        new.fcurves.remove(c)
    arm.animation_data.action = new
    for f, m in targets.items():
        sc.frame_set(f)
        pb.matrix = m
        bpy.context.view_layer.update()
        pb.keyframe_insert("location", frame=f, group="Bip01 Pelvis")
    return new, drift, (f1 - f0) / sc.render.fps
