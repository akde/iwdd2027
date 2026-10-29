"""Fetch MEVA videos (CC-BY-4.0, https://mevadata.org) from the public bucket mevadata-public-01.
python synth/scripts/fetch_meva.py list                 -> synth/data/meva/keys.csv (key, size, date, start, end, site, cam)
python synth/scripts/fetch_meva.py get KEYS.txt          -> synth/data/meva/videos/<file>.avi (skips existing)"""
import sys, re, pathlib, urllib.parse, concurrent.futures as cf
import requests, pandas as pd

BUCKET = "https://mevadata-public-01.s3.amazonaws.com/"
OUT = pathlib.Path(__file__).resolve().parents[1] / "data/meva"
PAT = re.compile(r"(\d{4}-\d\d-\d\d)\.(\d\d-\d\d-\d\d)\.(\d\d-\d\d-\d\d)\.([a-z\-]+)\.(G\d+)\.r13\.avi$")


def listing(prefix="drops-123-r13/"):
    rows, tok = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if tok:
            q["continuation-token"] = tok
        s = requests.get(BUCKET + "?" + urllib.parse.urlencode(q), timeout=60).text
        for k, z in re.findall(r"<Key>([^<]*)</Key>.*?<Size>(\d+)</Size>", s):
            m = PAT.search(k)
            if m:
                rows.append((k, int(z), *m.groups()))
        m = re.search(r"<NextContinuationToken>([^<]*)</NextContinuationToken>", s)
        if not m:
            break
        tok = m.group(1)
    df = pd.DataFrame(rows, columns=["key", "size", "date", "start", "end", "site", "cam"])
    OUT.mkdir(parents=True, exist_ok=True); df.to_csv(OUT / "keys.csv", index=False)
    print(len(df), "keys")


def get(key):
    dst = OUT / "videos" / key.split("/")[-1]
    if dst.exists():
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    with requests.get(BUCKET + key, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for c in r.iter_content(1 << 20):
                f.write(c)
    tmp.rename(dst)
    return dst.stat().st_size


if __name__ == "__main__":
    if sys.argv[1] == "list":
        listing()
    else:
        keys = [k.strip() for k in open(sys.argv[2]) if k.strip()]
        with cf.ThreadPoolExecutor(6) as ex:
            print("downloaded MB", sum(ex.map(get, keys)) / 1e6)
