import json
import requests
import time
import os
import sys
from tqdm import tqdm
import shutil
import numpy as np


def _load_json(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)


def _download(url, dst_file, skip_if_exists=True, verbose=True):
    if verbose:
        print("Downloading {} to {}.".format(url, dst_file))

    if skip_if_exists and os.path.isfile(dst_file):
        print("{} already exists. Skipped.".format(dst_file))
        return True

    dst_dir = os.path.dirname(dst_file)
    if not os.path.exists(dst_dir):
        if verbose:
            print("Creating directory {}".format(dst_dir))
        os.makedirs(dst_dir)

    start_time = time.time()
    try:
        r = requests.get(url, stream=True)
        if r.ok:
            num_bytes = 0
            with open(dst_file, 'wb') as f:
                for chunk in r.iter_content(32768):
                    num_bytes += len(chunk)
                    f.write(chunk)
            mbytes = num_bytes / float(1000000)
            elapsed_time = time.time() - start_time
            speed = mbytes / elapsed_time
            if verbose:
                print("Downloaded {:.2f}MB, speed {:.2f}MB/s.".format(
                    mbytes, speed))
            return True
        else:
            print("Download request failed.")
            return False
    except Exception as e:
        print("Download request failed with exception {}.".format(e))
        return False


# List of RGBD image ids.
rgbds = _load_json("rgbds.json")

_dst_path = "/path/to/Large_Dataset_Object_Scans"
_base_url = "https://redwood-3dscan.b-cdn.net"


def download_rgbd(scan_id, skip_if_exists=True, verbose=True):
    """Download RGBD scan by scan_id.
    Downloaded file will be saved to "data/rgbd/{scan_id}.zip".
    Args:
        scan_id: String of 5 digits, e.g. "00072".
        skip_if_exists: Skip downloading if the file already exists.
    """
    if scan_id in rgbds:
        url = "{}/rgbd/{}.zip".format(_base_url, scan_id)
        dst_file = os.path.join(_dst_path, "{}.zip".format(scan_id))
        _download(url, dst_file, skip_if_exists=skip_if_exists, verbose=verbose)
    else:
        print("RGBD scan_id {} is not available. Skipped.".format(scan_id))


def cleanup(scan_id, skip_if_exists=True):
    dst_file = os.path.join(_dst_path, "{}.zip".format(scan_id))
    dst_folder = os.path.join(_dst_path, "{}".format(scan_id))
    if skip_if_exists and (os.path.isdir(dst_folder) or not os.path.isfile(dst_file)):
        return
    try:
        os.mkdir(dst_folder)
        shutil.unpack_archive(dst_file, dst_folder)
        os.remove(dst_file)
        shutil.rmtree(os.path.join(dst_folder, "depth"))
        img_dir = os.path.join(dst_folder, "rgb")
        imgs = os.listdir(img_dir)
        imgs.sort()
        max_ind = int(imgs[-1].split("-")[0])
        if max_ind < 100:
            return
        keep_inds = np.linspace(1, max_ind, 100).astype(int)
        for img in imgs:
            ind = int(img.split("-")[0])
            if ind not in keep_inds:
                os.remove(os.path.join(img_dir, img))
    except Exception as e:
        print("CLEANUP FAILED")
        print(e)
        if os.path.isfile(dst_file):
            os.remove(dst_file)
        if os.path.isdir(dst_folder):
            shutil.rmtree(dst_folder)


if __name__ == "__main__":
    for id_ in tqdm(rgbds):
        download_rgbd(id_, verbose=False)
        cleanup(id_)
