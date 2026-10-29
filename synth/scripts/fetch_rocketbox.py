"""Fetch Microsoft Rocketbox files (MIT) at a pinned commit over HTTPS (they are plain git blobs, no LFS).
python synth/scripts/fetch_rocketbox.py PREFIX [PREFIX ...]   e.g. Assets/Animations Assets/Avatars/Adults
Files land in synth/assets/rocketbox/<repo path>; existing files with the right size are skipped."""
import sys, pathlib, json, concurrent.futures as cf
import requests

COMMIT = "0943055db6ec570bcef9f2c8b41c9e5467c808f9"  # master, 2022-10-02
ROOT = pathlib.Path(__file__).resolve().parents[1] / "assets/rocketbox"
TREE = f"https://api.github.com/repos/microsoft/Microsoft-Rocketbox/git/trees/{COMMIT}?recursive=1"
MEDIA = f"https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/{COMMIT}/"


def get(item):
    dst = ROOT / item["path"]
    if dst.exists() and dst.stat().st_size > 0:
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(MEDIA + requests.utils.quote(item["path"]), timeout=300)
    r.raise_for_status()
    if r.content.startswith(b"version https://git-lfs"):
        raise RuntimeError(f"got an LFS pointer for {item['path']}")
    tmp = dst.with_suffix(dst.suffix + ".part"); tmp.write_bytes(r.content); tmp.rename(dst)
    return len(r.content)


def main(prefixes):
    tree = requests.get(TREE, timeout=60).json()["tree"]
    items = [x for x in tree if x["type"] == "blob" and any(x["path"].startswith(p) for p in prefixes)]
    print(len(items), "files")
    with cf.ThreadPoolExecutor(8) as ex:
        n = sum(ex.map(get, items))
    print("downloaded MB", n / 1e6)


if __name__ == "__main__":
    main(sys.argv[1:])
