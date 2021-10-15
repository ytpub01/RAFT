from glob import glob
satimages = glob('datasets/asphere/satimages/*.png')
from PIL import Image
import tqdm as tq
import numpy as np

for imp in tq.tqdm(satimages):
    try:
        a = np.array(Image.open(imp))
    except:
        tq.tqdm.write(os.path.splitext(os.path.basename(imp)))

snaps = glob('datasets/asphere/snapshots/*.png')
snaps
for imp in tq.tqdm(snaps):
    try:
        a = np.array(Image.open(imp))
    except:
        tq.tqdm.write(os.path.splitext(os.path.basename(imp)))
