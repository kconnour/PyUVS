from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


files = sorted(Path('/mnt/science/data/mars/maven/iuvs/radiance/orbit03400/').glob('*orbit03400*'))
hduls = sorted(Path('/mnt/science/data_lake/mars/maven/iuvs/production/orbit03400').glob('*orbit03400*'))
file = files[0]
radiance = np.load(file) * 24/22
w = np.linspace(204, 305, num=19)
hdul = fits.open(hduls[0])
sza = hdul['pixelgeometry'].data['pixel_solar_zenith_angle'][-1, -1]

'''franck = np.array([0.989154E-01, 0.903521E-01,0.794510E-01,0.773823E-01, 0.691429E-01,0.652589E-01,0.582885E-01,
          0.568010E-01, 0.560545E-01, 0.526185E-01, 0.513500E-01, 0.516229E-01, 0.520269E-01, 0.513620E-01, 0.469003E-01])'''
l1cfiles = sorted(Path('/media/kyle/Athena/newl1c').glob('*orbit03400*'))
l1c = np.genfromtxt(l1cfiles[0], skip_header=547946)
franck = l1c[:, 2] * np.cos(np.radians(sza))

fig, ax = plt.subplots()
ax.plot(w, franck, color='r')
ax.plot(w, radiance[-1, -1, :], color='g')
#ax.plot(franck/radiance[-1, -1, :])

plt.savefig('/home/kyle/kylevsfranck.png')
