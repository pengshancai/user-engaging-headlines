import json
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data_dir", type=str, default='')
    parser.add_argument("--prefix", type=str, default='')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    fnames = [fname for fname in os.listdir(args.data_dir) if fname.startswith(args.prefix)]
    idx2kps = {}
    for fname in fnames:
        with open(args.data_dir + fname) as f:
            idx2kps_i = json.load(f)
            idx2kps = {**idx2kps_i, **idx2kps}
    with open(args.data_dir + args.prefix + '.json', 'w') as f:
        json.dump(idx2kps, f)




