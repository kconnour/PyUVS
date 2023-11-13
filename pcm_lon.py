import numpy as np
import matplotlib.pyplot as plt

lon_centers = np.linspace(-180, 180, num=65)
lon_boundaries = (lon_centers[1:] + lon_centers[:-1])/2
lon_boundaries = np.where(lon_boundaries<0, lon_boundaries + 360, lon_boundaries)
lon_boundaries = np.roll(lon_boundaries, 32)

lon_boundaries = np.concatenate(([0], lon_boundaries, [360]))
print(lon_boundaries)
pcm_Y = np.linspace(90+3.75/2, -90-3.75/2, num=50)

#raise SystemExit(9)

iuvs_pcm_dust = np.load(f'/mnt/science/data/mars/maven/iuvs/retrievals/gcm_grid/pcm/orbit03400/orbit03400-dust.npy')
iuvs_pcm_ice = np.load(f'/mnt/science/data/mars/maven/iuvs/retrievals/gcm_grid/pcm/orbit03400/orbit03400-ice.npy')

fig, ax = plt.subplots(1, 2)
ax[0].pcolormesh(lon_boundaries, pcm_Y, iuvs_pcm_dust)
ax[1].imshow(iuvs_pcm_ice)
plt.savefig('/home/kyle/pcmseam.png')
