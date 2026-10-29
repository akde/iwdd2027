"""Stage 3 (run inside Blender): blender -b -P synth/scripts/build_assets.py
Objects: each Poly Haven glTF (synth/assets/objects/<asset>/) -> one mesh object, origin at the bottom centre,
transforms applied (metres, Z up), saved in synth/assets/objects/library.blend; table synth/data/object_table.csv
(asset, object_type, size class, dims, licence). Avatars: every downloaded Rocketbox avatar imported once;
table synth/data/avatar_table.csv (avatar, gender, height_m, root_z, verts, textures_ok, worker)."""
import sys, pathlib, csv
import bpy, mathutils
SYN = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYN / "gen")); sys.path.insert(0, str(SYN / "scripts"))
import rb
from fetch_objects import ASSETS


def size_class(d):
    m = max(d)
    return "small" if m < 0.35 else "medium" if m < 0.8 else "large"


def build_objects():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    rows = []
    for asset, otype in ASSETS.items():
        gl = next((SYN / "assets/objects" / asset).glob("*.gltf"))
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=str(gl))
        new = [o for o in bpy.data.objects if o not in before]
        names = [o.name for o in new]
        meshes = [o for o in new if o.type == "MESH"]
        for o in new:
            o.select_set(o in meshes)
        bpy.context.view_layer.objects.active = meshes[0]
        for o in meshes:  # world transforms into the mesh data, then join
            o.parent = None if o.parent not in meshes else o.parent
        bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        if len(meshes) > 1:
            bpy.ops.object.join()
        ob = bpy.context.view_layer.objects.active
        for nm in names:  # joined meshes are gone; drop the remaining empties
            o = bpy.data.objects.get(nm)
            if o is not None and o is not ob:
                bpy.data.objects.remove(o)
        vs = [v.co for v in ob.data.vertices]
        lo = mathutils.Vector([min(v[i] for v in vs) for i in range(3)])
        hi = mathutils.Vector([max(v[i] for v in vs) for i in range(3)])
        off = mathutils.Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z))
        for v in ob.data.vertices:
            v.co -= off
        ob.location = (0, 0, 0); ob.name = asset; ob.data.name = asset
        d = hi - lo
        rows.append(dict(asset=asset, object_type=otype, size=size_class(d), dx=round(d.x, 3), dy=round(d.y, 3),
                         dz=round(d.z, 3), verts=len(ob.data.vertices), licence="CC0", source="polyhaven.com"))
        ob.use_fake_user = True
    bpy.ops.wm.save_as_mainfile(filepath=str(SYN / "assets/objects/library.blend"))
    write(SYN / "data/object_table.csv", rows)
    return rows


def build_avatars():
    rows = []
    for sub in ("Adults", "Professions"):
        for p in sorted((rb.RB / "Avatars" / sub).glob("*/Export/*.fbx")):
            name = p.parent.parent.name
            if p.stem != name:
                continue  # *_facial.fbx
            bpy.ops.wm.read_factory_settings(use_empty=True)
            arm = rb.import_avatar(name)
            meshes = [o for o in arm.children if o.type == "MESH"]
            zs = [(o.matrix_world @ v.co).z for o in meshes for v in o.data.vertices]
            tex_ok = all(n.image is not None and pathlib.Path(n.image.filepath).exists()
                         for o in meshes for m in o.data.materials for n in m.node_tree.nodes if n.type == "TEX_IMAGE")
            g = "f" if "female" in name.lower() else "m"
            rows.append(dict(avatar=name, group=sub, gender=g, height_m=round(max(zs) - min(zs), 3),
                             root_z=round(arm.location.z, 4), verts=sum(len(o.data.vertices) for o in meshes),
                             textures_ok=tex_ok, worker=sub == "Professions" and any(
                                 k in name for k in ("Construction", "Gardener", "Delivery", "Wood"))))
            print("AVATAR", rows[-1], flush=True)
    write(SYN / "data/avatar_table.csv", rows)
    return rows


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    o = build_objects()
    a = build_avatars()
    print("OBJECTS", len(o), "AVATARS", len(a))
