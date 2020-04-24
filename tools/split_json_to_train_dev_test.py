import json
import sys

if __name__ == "__main__":
    file_path = sys.argv[1]

    out_files = {
        "train": open(file_path + ".train", mode="w"),
        "valid": open(file_path + ".valid", mode="w"),
        "test": open(file_path + ".test", mode="w"),
    }

    with open(file_path, mode="r") as f_in:
        for line in f_in:
            json_item = json.loads(line.strip())
            out_files[json_item["set"]].write(json.dumps(json_item))
            out_files[json_item["set"]].write("\n")

    for k,v in out_files.items():
        v.close()

