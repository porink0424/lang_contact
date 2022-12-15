import argparse
import glob

def main(ids):
    res = {
        'train-not-ended': [],
        'organize-data-not-ended': [],
        'topsim-not-ended': [],
        'all-ended': [],
    }

    for id in ids:
        # check if train ended
        files = glob.glob(f"result/{id}*.txt")
        if len(files) <= 0:
            res['train-not-ended'].append(id)
            continue
        with open(files[0], "r") as f:
            raw_text = f.read()
            if "--------------------End--------------------" not in raw_text:
                res['train-not-ended'].append(id)
                continue
        
        # check if organize data ended
        files = glob.glob(f"result_md/{id}*.md")
        if len(files) <= 0:
            res['organize-data-not-ended'].append(id)
            continue

        # check if topsim ended
        with open(files[0], "r") as f:
            md_text = f.read()
            if "L_1_topsim_fill_me" in md_text:
                res['topsim-not-ended'].append(id)
                continue
        
        res['all-ended'].append(id)
    
    for key in ['train-not-ended', 'organize-data-not-ended', 'topsim-not-ended', 'all-ended']:
        print(key, *res[key])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ids', required=True, nargs="*", type=str)
    arg = parser.parse_args()

    print("----------------- RESULT -----------------")
    main(arg.ids)
