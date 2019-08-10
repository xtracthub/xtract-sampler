import os
import sys
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
from xtract_netcdf_main import extract_netcdf_metadata

os.chdir('xtract-sampler')
print(os.getcwd())
img_extensions = ["jpg", "png", "gif", "bmp", "jpeg", "tif", "tiff", "jif",
                  "jfif", "jp2", "jpx", "j2k", "j2c", "fpx", "pcd"]


class SystemReader(object):
    """Traverses file system, and produces initial dataset for prediction."""

    def __init__(self, top_dir):
        """Initializes SystemReader class.

        top_dir (str): The starting directory of files to get
        features from.
        feature_maker (class): An instance of the HeadBytes,
        RandBytes, RandHead class.
        """
        if not os.path.isdir(top_dir):
            raise NotADirectoryError("%s is not a valid directory" % top_dir)

        self.dirname = top_dir
        self.next_dirs = []
        self.filepaths = []

    def parse_dir(self, dirname):
        """Parse a directory with path dirname, add subdirectories to
        the list to be processed, and extract features from files.

        Parameter:
        dirname (str): Name of directory to parse.
        """
        for name in os.listdir(dirname):
            if name[0] == ".":
                continue
            if os.path.isfile(os.path.join(dirname, name)):
                self.filepaths.append(os.path.join(dirname, name))
            elif os.path.isdir(os.path.join(dirname, name)):
                self.next_dirs.append(os.path.join(dirname, name))

    def run(self):
        """Extract features from all files in top_dir."""
        self.next_dirs = [self.dirname]

        while self.next_dirs:
            dirname = self.next_dirs.pop(0)
            self.parse_dir(dirname)


def get_extension(filepath):
    ext = os.path.splitext(filepath)[-1].lower()
    return ext[1:len(ext)]


# TODO: Add a 'verbose' mode that actually prints the exceptions.
def infer_type(filepath):
    print(filepath)
    try:
        with timeout(seconds=15):
            if get_extension(filepath) in img_extensions:
                return "image"
            try:
                extract_netcdf_metadata(filepath)
                return "netcdf"
            except Exception as e:
                print("{} netcdf: {}".format(filepath, e))
                pass
            try:
                extract_json_metadata(filepath)
                return "json/xml"
            except Exception as e:
                print("{} jsonxml: {}".format(filepath, e))
                pass
            try:
                extract_columnar_metadata(filepath, parallel=False)
                return "tabular"
            except Exception as e:
                print("{} tabular: {}".format(filepath, e))
                pass
            try:
                if extract_keyword(filepath)["keywords"]:
                    return "freetext"
                else:
                    pass
            except Exception as e:
                print("{} freetext: {}".format(filepath, e))
                pass
            return "unknown"
    except TimeoutError:
        print("{} timed out".format(filepath))
        return "unknown"


def create_row(filepath):
    t0 = time.time()
    row = [filepath, os.path.getsize(filepath), infer_type(filepath)]
    row.append(time.time() - t0)
    print(row)
    return row


def write_naive_truth(outfile, top_dir, multiprocess=False):
    system_reader = SystemReader(top_dir)
    system_reader.run()
    print("There are {} files to be processed".format(len(system_reader.filepaths)))

    t0 = time.time()
    with open(outfile, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["path", "size", "file_label", "infer_time"])


        # TODO: Cut up the search space beforehand.
        if multiprocess:
            pools = mp.Pool()
            list_of_rows = pools.map(create_row, system_reader.filepaths)
            pools.close()
            pools.join()

            print("done getting rows")

            for row in list_of_rows:
                csv_writer.writerow(row)
        else:
            for filepath in system_reader.filepaths:
                csv_writer.writerow(create_row(filepath))
        csv_writer.writerow([time.time() - t0])


file_path = "/home/ryan/Documents/CS/CDAC/official_xtract/nist_dataset/"
t0 = time.time()
write_naive_truth("mdr_item_1051.csv", file_path, multiprocess=True)
print("total time: {}".format(time.time() - t0))


