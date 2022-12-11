import glob
import re
import argparse

def extract_other_topsim(ids):
    topsim_results = [
        [],[],[],[]
    ]
    for id in ids:
        files = glob.glob(f"result/{id}*.txt")
        with open(files[0], 'r') as f:
            raw_text = f.read()
        for i, match in enumerate(re.compile(r"RESULT TOPSIM: (.*?)\n").finditer(raw_text)):
            topsim_results[i].append(float(match.group(1)))
    print(topsim_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ids', required=True, nargs="*", type=str)
    args = parser.parse_args()

    extract_other_topsim(args.ids)