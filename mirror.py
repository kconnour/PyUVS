from pathlib import Path
from astropy.io import fits
import math
import matplotlib.pyplot as plt
import numpy as np

orbit = 6986

orbit_code = 'orbit' + f'{orbit}'.zfill(5)
block_code = 'orbit' + f'{math.floor(orbit / 100) * 100}'.zfill(5)

files = sorted(Path(f'/media/kyle/iuvs/production/{block_code}').glob(f'*apoapse*{orbit_code}*muv*.gz'))


'''hdul = fits.open(file)
foo = hdul['engineering'].data['mirror_pos']
bar = hdul['integration'].data['mirror_dn']

a = 1'''


for file in files[:4]:
    hdul = fits.open(file)
    print(file)
    #print(hdul['integration'].data['mirror_deg'])
    #print(hdul['pixelgeometry'].data['pixel_corner_mrh_alt'])
    foo = hdul['engineering'].data['mirror_pos']
    bar = hdul['integration'].data['mirror_dn']
    print(foo)
    print(bar)
    '''fig, ax = plt.subplots()
    primary = hdul['primary'].data
    print(primary.shape, np.amax(primary))
    ax.imshow(primary[..., 10])
    plt.savefig(f'/home/kyle/iuvsmirror/{file.stem}.png')
    plt.close(fig)'''
