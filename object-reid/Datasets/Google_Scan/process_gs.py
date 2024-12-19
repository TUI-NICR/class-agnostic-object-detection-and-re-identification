import os
import shutil

PATH = "/path/to/Google_Scan"

for zip_file in os.listdir(PATH):
    p = os.path.join(PATH, zip_file)
    if zip_file[-4:] != ".zip":
        if os.path.isfile(p):
            os.remove(p)
        continue
    name = zip_file.split(".zip")[0]
    pd = os.path.join(PATH, name)
    shutil.unpack_archive(p, pd)
    os.remove(p)
    for file in os.listdir(pd):
        p = os.path.join(pd, file)
        if os.path.isfile(p):
            os.remove(p)
        else:
            if not file == "thumbnails":
                shutil.rmtree(p)
