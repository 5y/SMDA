"""Download small public artifacts at immutable revisions and check SHA256."""
import argparse
import json
from pathlib import Path
import urllib.request
from smda.data import file_sha256


def download(manifest, output, keys=None):
    lock = json.loads(Path(manifest).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    keys = keys or [k for k, v in lock["artifacts"].items() if v.get("download_by_default")]
    paths = {}
    for key in keys:
        item = lock["artifacts"][key]
        target = output / item["local_filename"]
        if not target.exists() or file_sha256(target) != item["sha256"]:
            prefix = "datasets/" if item["repo_type"] == "dataset" else ""
            url = f"https://huggingface.co/{prefix}{item['repo_id']}/resolve/{item['revision']}/{item['filename']}"
            tmp = target.with_suffix(target.suffix + ".part")
            try:
                urllib.request.urlretrieve(url, tmp)
                if file_sha256(tmp) != item["sha256"]:
                    raise ValueError(f"Checksum mismatch for {key}.")
                tmp.replace(target)
            finally:
                tmp.unlink(missing_ok=True)
        paths[key] = str(target)
        print(key, "verified")
    return paths


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="configs/artifacts.lock.json")
    p.add_argument("--output", default="data/downloads")
    p.add_argument("--keys", nargs="+")
    a = p.parse_args()
    download(a.manifest, a.output, a.keys)
