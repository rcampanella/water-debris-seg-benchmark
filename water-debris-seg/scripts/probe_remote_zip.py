"""List a remote zip (HTTP range reads) and optionally extract named members."""
import io, sys, zipfile, json, requests

class HttpFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        r = requests.head(url, timeout=30); r.raise_for_status()
        self.size = int(r.headers["Content-Length"]); self.fetched = 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos
    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0: return 0
        r = requests.get(self.url, headers={"Range": f"bytes={self.pos}-{self.pos+n-1}"}, timeout=120)
        r.raise_for_status(); data = r.content; b[:len(data)] = data
        self.pos += len(data); self.fetched += len(data); return len(data)

url, out = sys.argv[1], sys.argv[2]
f = HttpFile(url)
z = zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 20))
infos = z.infolist()
names = [i.filename for i in infos]
summary = {"url": url, "zip_bytes": f.size, "n_members": len(infos),
           "first": names[:15], "last": names[-5:],
           "dirs": sorted({n.rsplit("/", 1)[0] for n in names if "/" in n})[:50],
           "non_image": [n for n in names if not n.lower().endswith((".png", ".jpg", ".bin", ".pcd"))][:50]}
for want in sys.argv[3:]:
    for n in names:
        if n.endswith(want):
            z.extract(n, out); summary.setdefault("extracted", []).append(n)
summary["bytes_fetched"] = f.fetched
json.dump(summary, open(f"{out}/{url.rsplit('/',2)[-2]}_{url.rsplit('/',1)[-1]}.json", "w"), indent=1)
print(json.dumps(summary, indent=1)[:3000])
