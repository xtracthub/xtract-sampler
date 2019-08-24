

import queues
import os

def crawl_dir(json_or_server="json", directory=None):
    r = {}
    i = 0
    sub_dirs = [x[0] for x in os.walk(directory)]
    try:
        os.makedirs(extracted_files_dir)
    except:
        pass

    for subdir in sub_dirs:
        files = os.walk(subdir).__next__()[2]
        if len(files) > 0:
            for item in files:
                file_path = item # os.path.join(subdir, item)  # TODO: Don't need this because same-level crawl?
                
                if json_or_server == "json":
                    print(file_path)
                    i += 1
                    print(i)
                    queues.put_on_queue({"file_path": str(file_path)})


    return r


if __name__ == "__main__":

    print("Starting preprocessing")
    crawl_dir(directory="/projects/DLHub/tyler/sampler_train_set")
