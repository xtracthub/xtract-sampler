import os
import sys
import argparse
import csv
import multiprocessing as mp
import time
from timeout import timeout
os.chdir('..')
sys.path.insert(0, 'xtract-jsonxml')
sys.path.insert(0, 'xtract-netcdf')
sys.path.insert(0, 'xtract-keyword')
sys.path.insert(0, 'xtract-tabular')
from xtract_tabular_main import extract_columnar_metadata
from xtract_jsonxml_main import extract_json_metadata
from xtract_keyword_main import extract_keyword
from xtract_netcdf_main import extract_netcdf

os.chdir('xtract-sampler')
img_extensions = ["jpg", "png", "gif", "bmp", "jpeg", "tif", "tiff", "jif",
                  "jfif", "jp2", "jpx", "j2k", "j2c", "fpx", "pcd"]

import queues  # Our queue class for talking to SQS. 


def get_extension(filepath):
    ext = os.path.splitext(filepath)[-1].lower()
    return ext[1:len(ext)]


# TODO: Add a 'verbose' mode that actually prints the exceptions.
def infer_type(filepath):
    with timeout(seconds=15):
        if get_extension(filepath) in img_extensions:
            return "image"
        try:
            extract_netcdf(filepath)
            return "netcdf"
        except Exception as e:
            if e.__class__ == TimeoutError:
                return "unknown"
            pass
        try:
            extract_json_metadata(filepath)
            return "json/xml"
        except Exception as e:
            if e.__class__ == TimeoutError:
                return "unknown"
            pass
        try:
            extract_columnar_metadata(filepath, parallel=False)
            return "tabular"
        except Exception as e:
            if e.__class__ == TimeoutError:
                return "unknown"
            pass
        try:
            if extract_keyword(filepath)["keywords"]:
                return "freetext"
            else:
                pass
        except Exception as e:
            if e.__class__ == TimeoutError:
                return "unknown"
            pass
        return "unknown"


def create_row(max_list):
    print("Hi")
    t0 = time.time()
    # TODO: TYLER Don't be deleting from queue until we're live. 
    filepath = "/projects/DLHub/tyler/sampler_train_set/" + queues.pull_off_queue()["file_path"]
    print(filepath)
    row = {"file_path": filepath, "file_size": os.path.getsize(filepath), "sample_type": infer_type(filepath), "total_time": time.time() - t0}

    # TODO: Send to results queue here. 
    queues.put_on_results_queue(row)
    print("File successfully processed!") 

    return row


def write_naive_truth(outfile, top_dir, multiprocess=False, chunksize=1, n=1000):
    t0 = time.time()

    print(multiprocess)
    if multiprocess:
        pools = mp.Pool()
        max_list = [0] * 110639  # TODO: Add a function in 'queues' for this.
        run_par =  pools.imap_unordered(create_row, max_list, chunksize=chunksize)

        pools.close()
        pools.join()

    print("Automated training time: {}".format(time.time() - t0))

