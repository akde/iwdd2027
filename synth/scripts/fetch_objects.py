"""Stage 3: fetch CC0 object meshes from Poly Haven (glTF, 1k textures) -> synth/assets/objects/<asset>/.
python synth/scripts/fetch_objects.py            (the list below; existing files are skipped)
Licence: CC0 (https://polyhaven.com/license); build_assets.py records licence and source per asset in synth/data/object_table.csv."""
import pathlib, concurrent.futures as cf
import requests

OUT = pathlib.Path(__file__).resolve().parents[1] / "assets/objects"
UA = {"User-Agent": "iwdd2027-synth-fetch-objects"}  # Poly Haven API terms: name the software
# object_type (taxonomy vocabulary) per asset; phase 2 furniture included so the library is built once
ASSETS = {
    "trashbag": "garbage_bag", "compost_bag_02": "sack", "cement_bag": "sack", "compost_bags": "many_bags",
    "cardboard_box_01": "cardboard_box", "plastic_crate_01": "box", "plastic_crate_03": "box",
    "wooden_crate_01": "box", "plastic_jerrycan": "other:jerrycan", "metal_jerrycan": "other:jerrycan",
    "old_tyre": "tire", "plastic_bottle_gallon": "bottle", "bleach_bottle": "bottle", "multi_cleaner_bottle": "bottle",
    "can_rusted": "can", "wicker_basket_01": "other:basket", "industrial_pastic_container": "box",
    "plastic_monobloc_chair_01": "furniture", "ArmChair_01": "furniture", "Sofa_01": "furniture",
    "SchoolChair_01": "furniture", "drawer_cabinet": "furniture", "painted_wooden_chair_01": "furniture",
}


def fetch(asset):
    files = requests.get(f"https://api.polyhaven.com/files/{asset}", headers=UA, timeout=60).json()
    g = files["gltf"]["1k"]["gltf"]
    todo = [(g["url"], OUT / asset / pathlib.Path(g["url"]).name)]
    todo += [(v["url"], OUT / asset / k) for k, v in g.get("include", {}).items()]
    n = 0
    for url, dst in todo:
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(url, headers=UA, timeout=300); r.raise_for_status()
        dst.write_bytes(r.content); n += len(r.content)
    return asset, n


if __name__ == "__main__":
    with cf.ThreadPoolExecutor(6) as ex:
        for a, n in ex.map(fetch, ASSETS):
            print(a, round(n / 1e6, 1), "MB")
