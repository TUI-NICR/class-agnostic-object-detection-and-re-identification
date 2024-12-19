import csv
import requests
import os
from tqdm import tqdm
import shutil

DEST = "/path/to/CO3D"


def download_with_progress_bar(url: str, fname: str, filename: str):
    # taken from https://stackoverflow.com/a/62113293/986477
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


with open("/path/to/object-reid/Misc/co3d_download_links.csv", "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    links = [r for r in reader]

for name, link in links:
    try:
        f_name = os.path.join(DEST, name)
        download_with_progress_bar(link, f_name, name)
        dest_dir = f_name.split(".zip")[0]
        shutil.unpack_archive(f_name, dest_dir)
    except Exception as e:
        print("DOWNLOAD HAD A STROKE!\n")
        print(e)
        os.remove(f_name)
        continue
    os.remove(f_name)
    d = os.listdir(dest_dir)[0]
    p = os.path.join(dest_dir, d)
    for dd in os.listdir(p):
        p = os.path.join(dest_dir, d, dd)
        if os.path.isdir(p):
            if len(os.listdir(p)) == 0:
                os.rmdir(p)
            else:
                for ddd in os.listdir(p):
                    p = os.path.join(dest_dir, d, dd, ddd)
                    if ddd not in ["images", "masks"]:
                        if os.path.isdir(p):
                            shutil.rmtree(p)
                        else:
                            os.remove(p)
