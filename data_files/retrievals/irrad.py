from netCDF4 import Dataset
from datetime import datetime
from astropy.time import Time, TimeDelta
import numpy as np


dt = datetime(2020, 4, 13)

dataset = Dataset('/mnt/science/data_lake/sun/tsis-1/tsis_ssi_L3_c24h_latest.nc')
jd = Time(str(dt), format='iso').jd
times = dataset['time'][:]
idx = np.abs(times - jd).argmin()
irr = dataset['irradiance'][idx, :].data
print(irr)
