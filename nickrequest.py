from astropy.io import fits
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

p = sorted(Path('/media/kyle/iuvs/production/orbit07800').glob('*apoapse*orbit07851*muv*'))

# Orbit 7851
for files in p:
    hdul = fits.open(files)
    primary = hdul['primary'].data
    if primary.shape[-1] != 19:
        continue
    print(np.amax(primary[..., -2]))
