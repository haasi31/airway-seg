import os
import glob
from multiprocessing import Pool
from tqdm import tqdm


"""
Predict the 5 lobe masks of a lung to create the simulation spaces.
Utilizing https://github.com/JoHof/lungmask
"""

def process_file(path):
    filename = os.path.basename(path)
    filename = f"{filename.split('.')[0]}_lobes_mask.nii.gz"
    output_path = os.path.join("/home/shared/Data/ATM22/train/masks", filename)
    if os.path.exists(output_path):
        print(f"Skipping {path}, already exists")
        return
    os.system(f"lungmask --modelname LTRCLobes_R231 --batchsize 16 {path} {output_path} --noprogress >/dev/null 2>&1")

if __name__ == "__main__":
    input_dir = "/home/shared/Data/ATM22/train/images"
    files = glob.glob(os.path.join(input_dir, "*"))

    with Pool(5) as p, tqdm(total=len(files), desc='Lobe masks in progress...') as pbar:
        for _ in p.imap_unordered(process_file, files):
            pbar.update()