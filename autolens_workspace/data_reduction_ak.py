import numpy as np
from astropy.io import fits
from astropy.convolution import convolve
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.visualization import astropy_mpl_style
from astropy.utils.data import get_pkg_data_filename
plt.style.use(astropy_mpl_style)

from pyprojroot import here
from os import path


#Visualization
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)

workspace_path = str(here())
dataset_path = path.join(workspace_path,"474_project", "474_data", "ak\\")

class data_reduction(object):
    '''
    Class data_reduction gives the final fits file after centering, cropping and finally convolving
    '''

    def __init__(self, filepath='filepath', filename='filename', RA='', DEC=''):
        '''
        runs when the class is called

        Inputs
        filepath: /path/to/file/
        filename: name of file
        ra: Right ascension of object of interest
        dec: declination of object of interest
        '''

        print('Reading the fits file.')

        self.filepath=filepath
        self.filename=filename

        self.hdul=fits.open(self.filepath+self.filename+'.fits')
            #self.hdul.verify('fix')

        #Choosing the data header, usually header[1]
        self.data = self.hdul[1].data

        #World co-ordinates of image
        self.wcs = WCS(fobj=self.hdul[1], header=self.hdul[1].header)

        #Weight and context of image
        self.weight = self.hdul[2].data
        self.context = self.hdul[3].data

        self.ra = RA
        self.dec = DEC

        print('Input RA:', self.ra, 'Input DEC:', self.dec)

    def HMS2deg(self, ra='', dec=''):
        '''
        Function converts RA and DEC of object to degrees

        Inputs
        ra: Right ascension of object of interest
        dec: declination of object of interest

        Outputs
        RA: Right Ascension in degrees
        DEC: Declination in degrees
        '''
        RA, DEC, rs, ds = '', '', 1, 1
        if dec:
            D, M, S = [float(i) for i in dec.split()]
            if str(D)[0] == '-':
                ds, D = -1, abs(D)
            deg = D + (M/60) + (S/3600)
            DEC = '{0}'.format(deg*ds)

        if ra:
            H, M, S = [float(i) for i in ra.split()]
            if str(H)[0] == '-':
                rs, H = -1, abs(H)
            deg = (H*15) + (M/4) + (S/240)
            RA = '{0}'.format(deg*rs)
        #TODO: RA with value greater than 180 degrees should be subtracted by 360.(check)

        if ra and dec:
            return (float(RA), float(DEC))
        else:
            return float(RA) or float(DEC)

    def center(self):
        '''
        Function to find the centre of the image

        Outputs
        x_center: pixel x coordinate of centre of object of interst
        y_center: pixel y coordinate of centre of object of interst
        '''

        #updates ra and dec
        self.ra, self.dec = self.HMS2deg(ra=self.ra, dec=self.dec)
        print('New RA in degrees:', self.ra, 'New DEC in degrees:',self.dec)

        #Center coordinates of the object of interest
        x_center, y_center = self.wcs.all_world2pix(self.ra, self.dec, 1, adaptive=False, ra_dec_order=True)

        print('Center coords:', x_center, y_center)

        return x_center, y_center

    def crop(self, cropsize=100):
        '''
        Function to crop the data

        Input
        cropsize: The size of a single axis, delfault value is 74 (This value is chosen because the PSF has 74x74 dimensions)
        outfilepath: /path/to/save/outfile/in/

        Output
        Creates an output .fits file
        '''

        self.xc, self.yc = self.center()

        #Indices of array can only be integers, Brfore converting float to int, round the decimal to the closest value
        xc = int(np.round(self.xc))
        yc = int(np.round(self.yc))
        cs = int(np.round(cropsize/2))
        print('Rounded pixel coord center values:', xc, yc)

        #Defining boundaries of pixel coords of the image
        x1 = xc-cs
        x2 = xc+cs

        y1 = yc-cs
        y2 = yc+cs

        self.hdul[1].data=self.data[y1:y2,x1:x2]
        #cropping weight and context to reduce filesize
        self.hdul[2].data=self.weight[y1:y2,x1:x2]
        self.hdul[3].data=self.context[y1:y2,x1:x2]

        return

    def noise_map(self):
        '''
        Function generates a noise map
        '''

        mean_snr = self.hdul['SCI'].header['SNRMEAN']
        poisson_dist = self.hdul[1].data/np.random.poisson((self.hdul['SCI'].header['SNRMEAN']), self.hdul[1].data.shape)

        return poisson_dist

    def convolve(self, psffile='psffile'):
        '''
        Function convolves the HST image with PSF

        Inputs
        psffile: /path/to/psffile/psffilename
        '''

        self.image_data = self.hdul[1].data

        with fits.open(psffile) as self.psf_hdul:
            #data for psf in header[0]
            self.psf_data = self.psf_hdul[0].data

        #kernel should be odd for convolution
        self.convolved_data=convolve(self.image_data[0:73, 0:73], self.psf_data[0:73, 0:73])

        return

    def writefile(self, outfilepath='outfilepath'):
        '''
        Function writes to the fits file

        1) Append the convolved image
        2) Overwrite the cropped data
        '''
        self.noise_map_data=self.noise_map()
        self.image_data=self.hdul[1].data

        #appending noise map to the fits file
        self.noise_map_ImageHDU = fits.ImageHDU(name='NMAP', data=self.noise_map_data)
        self.hdul.append(self.noise_map_ImageHDU)

        #appending PSF to the fits file
        self.psf_ImageHDU = fits.ImageHDU(name='PSF', data=self.psf_data)
        self.hdul.append(self.psf_ImageHDU)

        #appending Covolve to the fits file
        self.convolve_ImageHDU = fits.ImageHDU(name='CONVOLVE', data=self.convolved_data)
        self.hdul.append(self.convolve_ImageHDU)

        self.hdul.writeto(outfilepath + self.filename + '_cropped_convolved.fits',overwrite=True)

        hdu_image = fits.PrimaryHDU(data=self.image_data)
        hdu_image.writeto(outfilepath+'imaging.fits')

        hdu_noise = fits.PrimaryHDU(data=self.noise_map_data)
        hdu_noise.writeto(outfilepath+'noise_map.fits')

        hdu_psf = fits.PrimaryHDU(data=self.psf_data)
        hdu_psf.writeto(outfilepath+'psf.fits')

        self.hdul.close()

        return

    def plot_images(self, cropped_image=True, noise_map=True, psf=True, convolve=True, outfilepath='outfilepath'):
        '''
        plot images

        cropped_image produces plot of the centered and cropped image data.
        noise_map produces a noise map plot for the cropped image
        psf produces plot of the Point Spread Function.
        convolved produces plot of data from cropped image data and psf covolved.
        '''

        self.outfilepath=outfilepath

        if cropped_image == True:
            plt.figure()
            plt.imshow(self.image_data, cmap='gray', norm=LogNorm())
            plt.colorbar()
            plt.title('Image', fontsize=20)
            plt.savefig(self.outfilepath+'cropped_image.pdf', dpi=200)
            plt.show()
            plt.clf()
        else:
            print('Plotting cropped image is turned off')

        if psf == True:
            plt.figure()
            plt.imshow(self.psf_data, cmap='gray', norm=LogNorm())
            plt.colorbar()
            plt.title('PSF', fontsize=20)
            plt.savefig(self.outfilepath+'psf.pdf', dpi=200)
            plt.show()
            plt.clf()
        else:
            print('Plotting PSF is turned off')

        if noise_map == True:
            plt.figure()
            plt.imshow(self.noise_map_data, cmap='gray', norm=LogNorm())
            plt.colorbar()
            plt.title('Noise Map', fontsize=20)
            plt.savefig(self.outfilepath+'noise_map.pdf', dpi=200)
            plt.show()
            plt.clf()

            #Plotting signal + noisemap
            plt.figure()
            plt.imshow(self.image_data+self.noise_map_data, cmap='gray', norm=LogNorm())
            plt.colorbar()
            plt.title('Signal+Noise', fontsize=20)
            plt.savefig(self.outfilepath+'signal_plus_noise_map.pdf', dpi=200)
            plt.show()
            plt.clf()
        else:
            print('Plotting PSF is turned off')

        if convolve == True:
            plt.figure()
            plt.imshow(self.convolved_data, cmap='gray', norm=LogNorm())
            plt.colorbar()
            plt.title('Convoluted Image', fontsize=20)
            plt.savefig(self.outfilepath+'convolve.pdf', dpi=200)
            plt.show()
            plt.show()
        else:
            print('Plotting convolved image is turned off')

        return


reduced_data = data_reduction(filepath=dataset_path,filename="j9op04010_drc_cropped", RA='02h 52045.2100"', DEC='+00◦3905800"')
reduced_data.writefile('test_reduce')