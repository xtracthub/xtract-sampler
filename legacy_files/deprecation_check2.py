import json
import random
import math


path = '/home/tskluzac/xtract-crawler/globus/result.json'

def load_json(json_path):
    with open(json_path, 'r') as f:
        crawled_json = json.load(f)

    return crawled_json

def random_selection(json_file):
    total_size = 0
    num_compressed = 0
    compressed_size = 0

    crawled_json  = json_file

    crawled_json_keys = list(crawled_json.keys())
    crawled_json_values = list(crawled_json.values())
    print("Done getting values")
    new_json_size = math.floor(len(crawled_json) * 0.05)
    new_json_indices = random.sample(range(len(crawled_json)), new_json_size)

    new_json = {}
    for indice in new_json_indices:
        # indice = indice.replace('//', '/')
        total_size += crawled_json_values[indice]['physical']['size']

        if crawled_json_values[indice]['physical']['extension'] in ["zip", "tar", "gz", "tgz", "Z"]:
            num_compressed += 1
            compressed_size += crawled_json_values[indice]['physical']['size']

        new_json.update({crawled_json_keys[indice]: crawled_json_values[indice]})

    print("Total size: {} TB".format(total_size / (10 ** 12)))
    print("Num compressed: {} files".format(num_compressed))
    print("Compressed Size: {} GB".format(compressed_size / (10 ** 9)))

    total_size = total_size/(10 ** 12)
    print(total_size)
    if total_size < 0.45:
        print("YAY!")
        return new_json
    
with open('rand_samp.json','w') as g:
    rand_sel = random_selection(load_json(path))
    json.dump(rand_sel, g)
    print("Successfully written to file!")
