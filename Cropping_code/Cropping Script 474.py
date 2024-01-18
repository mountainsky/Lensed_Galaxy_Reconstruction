# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 23:38:05 2021

@author: kogra
"""

from astropy.nddata.utils import Cutout2D as cut2D
from astropy.io import fits

#Visualization
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)

x1 = 2140
x2 = 2240

y1 = 3128
y2 = 3228

#x1:x2, y1:y2 represents the pixel cropping coordinates

filename = 'j9op04010_drc'
file = filename + '.fits'

hdul = fits.open(file)

data = hdul[1].data


hdul[1].data= data[y1:y2,x1:x2]

data_c = data[y1:y2,x1:x2]
#Note that data_c is not used explicitly as data,
#   just to show visuals of the new data

#print(data_c)

hdul.writeto(filename + '_cropped_new.fits',overwrite=True)

hdul.close()

plt.figure()
plt.imshow(data_c, cmap='gray',origin='lower')
plt.colorbar()
plt.savefig(filename + '_cropped_image.jpeg')
plt.show()
plt.close()

