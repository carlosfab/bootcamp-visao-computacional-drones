#!/usr/bin/env python3
"""Read a public ZIP through bounded HTTP ranges, without its full payload."""
import io
import json
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

URL = "https://zenodo.org/api/records/11048412/files/Cattle_drone_images_042024.zip/content"
SIZE = 16586848207
OUT = Path(__file__).resolve().parents[1] / "dados" / ".auditoria-fonte"
OUT.mkdir(parents=True, exist_ok=True)


class RangeReader(io.RawIOBase):
    def __init__(self, url=URL, size=SIZE):
        self.url, self.size, self.pos = url, size, 0
        self.cache = {}
        self.block = 1024 * 1024
        self.transferred = 0
        self.requests = 0
        self.cache_dir = OUT / "range-cache"
        self.cache_dir.mkdir(exist_ok=True)

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset + (self.pos if whence == 1 else self.size if whence == 2 else 0)
        if self.pos < 0:
            raise ValueError("negative seek")
        return self.pos

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        end = min(self.size, self.pos + n)
        if end <= self.pos:
            return b""
        if n > 32 * 1024 * 1024:
            raise ValueError("Refusing unexpectedly large read")
        chunks = []
        while self.pos < end:
            block_id = self.pos // self.block
            start = block_id * self.block
            stop = min(self.size, start + self.block)
            if block_id not in self.cache:
                cached = self.cache_dir / f"{block_id}.bin"
                if cached.exists() and cached.stat().st_size == stop - start:
                    self.cache[block_id] = cached.read_bytes()
                    continue
                result = subprocess.run([
                    "curl", "-fsSL", "--max-time", "60", "--max-filesize", str(self.block),
                    "--range", f"{start}-{stop - 1}", "--write-out", "%{http_code}", self.url,
                ], check=True, capture_output=True)
                raw, status = result.stdout[:-3], result.stdout[-3:]
                if status != b"206" or len(raw) != stop - start:
                    raise RuntimeError(f"Range rejected: HTTP {status!r}, bytes {len(raw)}, wanted {stop-start}")
                self.cache[block_id] = raw
                cached.write_bytes(raw)
                self.transferred += len(raw)
                self.requests += 1
                print(f"Fetched range {start}-{stop-1} ({self.requests} requests)", flush=True)
            take = min(end, stop) - self.pos
            offset = self.pos - start
            chunks.append(self.cache[block_id][offset:offset + take])
            self.pos += take
        return b"".join(chunks)


if __name__ == "__main__":
    reader = RangeReader()
    with zipfile.ZipFile(reader) as archive:
        rows = [{"path": item.filename, "size": item.file_size, "compressed_size": item.compress_size,
                 "header_offset": item.header_offset, "crc32": f"{item.CRC:08x}"}
                for item in archive.infolist() if not item.is_dir()]
        (OUT / "icaerus_inventory.json").write_text(json.dumps(rows, indent=2))
        images = [r for r in rows if r["path"].lower().endswith((".jpg", ".jpeg"))]
        groups = Counter("/".join(Path(r["path"]).parts[:-1]) for r in images)
        print(json.dumps({"images":len(images), "files":len(rows), "flights":len(groups),
                          "requests":reader.requests,"bytes_transferred":reader.transferred}, indent=2))
