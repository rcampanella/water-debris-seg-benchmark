import io, zipfile, requests
class HttpFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        for k in range(6):
            try: r = requests.head(url, timeout=30); break
            except requests.exceptions.ConnectionError:
                import time; time.sleep(30)
        r.raise_for_status()
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
        for k in range(6):
            try:
                r = requests.get(self.url, headers={"Range": f"bytes={self.pos}-{self.pos+n-1}"}, timeout=120)
            except requests.exceptions.ConnectionError:
                import time; time.sleep(30); continue
            if r.status_code != 429: break
            import time; time.sleep(float(r.headers.get("Retry-After", 60)))
        r.raise_for_status(); d = r.content; b[:len(d)] = d
        self.pos += len(d); self.fetched += len(d); return len(d)
def open_zip(url, buf=1<<16):
    return zipfile.ZipFile(io.BufferedReader(HttpFile(url), buffer_size=buf))
