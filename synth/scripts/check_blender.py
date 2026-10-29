"""Stage-0 check (run inside Blender): blender -b -P synth/scripts/check_blender.py -- OUT_DIR
Cycles on GPU (OptiX, else CUDA), 1280x720, a cube on a shadow-catcher plane, transparent film.
Writes OUT_DIR/cube_rgba.png and OUT_DIR/cube_shadow0001.png (Shadow Catcher pass, 16-bit, 1 = no shadow) and prints the devices used."""
import sys, pathlib
import bpy

out = pathlib.Path(sys.argv[sys.argv.index("--") + 1]); out.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "CYCLES"
prefs = bpy.context.preferences.addons["cycles"].preferences
for kind in ("OPTIX", "CUDA"):
    try:
        prefs.compute_device_type = kind
    except TypeError:
        continue
    prefs.get_devices()
    gpus = [d for d in prefs.devices if d.type == kind]
    if gpus:
        break
for d in prefs.devices:
    d.use = d.type == prefs.compute_device_type
print("DEVICES", prefs.compute_device_type, [(d.name, d.type, d.use) for d in prefs.devices])
sc.cycles.device = "GPU"
sc.cycles.samples = 32
sc.cycles.use_denoising = True
sc.render.resolution_x, sc.render.resolution_y = 1280, 720
sc.render.film_transparent = True
sc.view_settings.view_transform = "Standard"  # passes stay linear-ish: shadow catcher = 1 where unshadowed

bpy.ops.mesh.primitive_plane_add(size=20)
bpy.context.object.is_shadow_catcher = True
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.5))
bpy.ops.object.light_add(type="SUN", location=(0, 0, 10), rotation=(0.6, 0.3, 0.8))
bpy.context.object.data.energy = 4
world = bpy.data.worlds.new("w"); sc.world = world
world.use_nodes = True; world.node_tree.nodes["Background"].inputs[1].default_value = 0.3
bpy.ops.object.camera_add(location=(6, -6, 4), rotation=(1.1, 0, 0.785))
sc.camera = bpy.context.object

vl = sc.view_layers[0]
vl.cycles.use_pass_shadow_catcher = True
sc.use_nodes = True
nt = sc.node_tree
rl = nt.nodes["Render Layers"]
fo = nt.nodes.new("CompositorNodeOutputFile")
fo.base_path = str(out); fo.format.file_format = "PNG"; fo.format.color_depth = "16"; fo.format.color_mode = "RGB"
fo.file_slots[0].path = "cube_shadow"
nt.links.new(rl.outputs["Shadow Catcher"], fo.inputs[0])
sc.render.filepath = str(out / "cube_rgba.png")
sc.render.image_settings.color_mode = "RGBA"
bpy.ops.render.render(write_still=True)
print("DONE", out)
