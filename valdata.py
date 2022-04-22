import os
from glob import glob
from PIL import Image
import tqdm as tq
import numpy as np

satimages = glob('datasets/asphere/satimages/*.png')
for imp in tq.tqdm(satimages):
    try:
        a = np.array(Image.open(imp))
    except:
        print("Exception sat images")
        tq.tqdm.write(os.path.splitext(os.path.basename(imp)))

snaps = glob('datasets/asphere/snapshots/*.png')
for imp in tq.tqdm(snaps):
    try:
        a = np.array(Image.open(imp))
    except:
        print("Exception snap images")
        tq.tqdm.write(os.path.splitext(os.path.basename(imp)))
