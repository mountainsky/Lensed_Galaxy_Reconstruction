# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 09:10:01 2021

@author: kogra
"""
#from photutils.segmentation import detect_threshold
from astropy.stats import sigma_clipped_stats
from astropy.nddata import NDData
from astropy.table import Table
from photutils.detection import find_peaks
from astropy.io import fits
from photutils.psf import extract_stars
from photutils.psf import EPSFBuilder

#Visualization
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
from astropy.visualization import simple_norm
plt.style.use(astropy_mpl_style)


filename = 'j9op04010_drc'
file = filename + '.fits'

hdul = fits.open(file)

data = hdul[1].data

hdul.close()

#thresh_img=detect_threshold(data,3)
#print(thresh_img)

peaks_tbl = find_peaks(data, threshold=170.)  
peaks_tbl['peak_value'].info.format = '%.8g'  # for consistent table output  
print(peaks_tbl)  

size = 25
hsize = (size - 1) / 2
x = peaks_tbl['x_peak']  
y = peaks_tbl['y_peak']  
mask = ((x > hsize) & (x < (data.shape[1] -1 - hsize)) & \
        (y > hsize) & (y < (data.shape[0] -1 - hsize))) 

stars_tbl = Table()
stars_tbl['x'] = x[mask]  
stars_tbl['y'] = y[mask]

stars_tbl[]
    
nddata = NDData(data=data) 
stars = extract_stars(nddata, stars_tbl, size=25)

nrows = 5
ncols = 5
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20), squeeze=True)
ax = ax.ravel()
for i in range(nrows * ncols):
    norm = simple_norm(stars[i], 'log', percent=99.)
    ax[i].imshow(stars[i], norm=norm, origin='lower', cmap='viridis')
    
plt.show()

epsf_builder = EPSFBuilder(oversampling=4, maxiters=5)  
epsf, fitted_stars = epsf_builder(stars)  

norm = simple_norm(epsf.data, 'log', percent=99.)
plt.imshow(epsf.data, norm=norm, origin='lower', cmap='viridis')
plt.colorbar()
plt.show()
# =============================================================================
# plt.figure()
# plt.imshow(data_c, cmap='gray',origin='lower')
# plt.colorbar()
# plt.savefig(filename + '_cropped_image.jpeg')
# plt.show()
# plt.close()
# 
# =============================================================================
