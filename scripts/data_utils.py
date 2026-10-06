"""Streaming, resumable downloads; no executable content or archive extraction."""
import hashlib
import json
import time
import urllib.request
from pathlib import Path


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def download(url, path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    receipt=path.with_suffix(path.suffix+".source.json")
    if path.exists():
        if not receipt.exists() or json.loads(receipt.read_text())["sha256"] != sha256(path):
            raise ValueError(f"File exists without a matching receipt: {path}")
        return path
    part=path.with_suffix(path.suffix+".part")
    for attempt in range(3):
        try:
            offset=part.stat().st_size if part.exists() else 0
            headers={"User-Agent":"ZoneGuard-academic/1.0"}
            if offset:
                headers["Range"]=f"bytes={offset}-"
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=60) as response:
                resumed=response.status==206
                if not resumed:
                    offset=0
                expected=int(response.headers.get("Content-Length",0))+offset
                done=offset;next_log=done
                with part.open("ab" if resumed else "wb") as f:
                    while chunk:=response.read(1024*1024):
                        f.write(chunk);done+=len(chunk)
                        if done>=next_log:
                            print(f"{path.name}: {done//1024**2}/{expected//1024**2} MB",flush=True)
                            next_log=done+100*1024**2
                if expected and done!=expected:
                    raise OSError("Download ended early")
            part.replace(path)
            receipt.write_text(json.dumps({"url":url,"bytes":path.stat().st_size,"sha256":sha256(path)},indent=2),encoding="utf-8")
            return path
        except OSError:
            if attempt==2:
                raise
            time.sleep(2)


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url");parser.add_argument("path",type=Path)
    args=parser.parse_args();download(args.url,args.path)
