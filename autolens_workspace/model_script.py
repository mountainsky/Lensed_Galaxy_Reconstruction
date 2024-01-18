from pyprojroot import here
from os import path
import autolens as al
import autolens.plot as aplt
from astropy.io import fits
import autofit as af
import numpy as np

#Visualization
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)

workspace_path = str(here())
dataset_path = path.join(workspace_path,"474_project", "474_data", "nightingale","slacs_new","slacs0252+0039")

# =============================================================================
# psf_file = path.join(dataset_path, "psf.fits")
# hdul = fits.open(psf_file)
# #Could add in code later for automation to check first if data dimensions are odd or even, but easy enough to do manually
# psf_data = hdul[0].data[:-1,:-1] #PSF kernel must be odd for autolens to work
# hdul[0].data=psf_data
# hdul.writeto(path.join(dataset_path, 'PSF_model.fits'),overwrite=True)
# hdul.close()
# 
# =============================================================================

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "F814W_image.fits"),
    noise_map_path=path.join(dataset_path, "F814W_noise_map.fits"),
    psf_path=path.join(dataset_path, "F814W_psf.fits"),
    pixel_scales=0.05,
    image_hdu=0, 
    noise_map_hdu=0,
    psf_hdu=0
)

# =============================================================================
# plt.figure()
# plt.imshow(psf_data, cmap='gray',origin='lower')
# plt.colorbar()
# plt.show()
# plt.close()
# 
# =============================================================================

imaging_plotter = aplt.ImagingPlotter(imaging=imaging)
imaging_plotter.subplot_imaging()

mask = al.Mask2D.circular(
    shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=1.6)

#print(mask)  # 1 = True, which means the pixel is masked. Edge pixels are indeed masked.

visuals_2d = aplt.Visuals2D(mask=mask)
imaging_plotter = aplt.ImagingPlotter(imaging=imaging, visuals_2d=visuals_2d)
imaging_plotter.figures_2d(image=True)

imaging = imaging.apply_mask(mask=mask)

include_2d = aplt.Include2D(mask=True)
imaging_plotter = aplt.ImagingPlotter(imaging=imaging, include_2d=include_2d)
imaging_plotter.figures_2d(image=True)

e_comps = al.convert.elliptical_comps_from(axis_ratio=0.93, angle=106.2)
print(f"Ellipticity components: {e_comps}")

#Plotting positions of source images just for testing to make nonlinear search faster
#Might need to update positions still, not sure the specifics on how the model constrains it 
# =============================================================================
# positions1 = al.Grid2DIrregular(grid=[(0.0, 0.9), (1.1, 0.0), (0.0, -1.1), (-0.9, 0.0)])
# #positions1 is what I previously guessed for image positions
# visuals_2d = aplt.Visuals2D(positions=positions1)
# imaging_plotter = aplt.ImagingPlotter(imaging=imaging, visuals_2d=visuals_2d)
# imaging_plotter.subplot_imaging()
# 
# =============================================================================
positions2 = [[-0.925, 0.025], [-0.875, 0.225], [1.075, 0.275], [0.9750000000000001, -0.5750000000000001]]
#Positions2 is what I believe Nightingale et al. used in their report
visuals_2d = aplt.Visuals2D(positions=positions2)
imaging_plotter = aplt.ImagingPlotter(imaging=imaging, visuals_2d=visuals_2d)
imaging_plotter.subplot_imaging()


# =============================================================================
# #Not sure where to find the shearing elliptical comps
# shear_info = al.convert.shear_magnitude_and_angle_from(e_comps)
# print(f"Shear magnitude and angle: {shear_info}")
# 
# =============================================================================

lens_galaxy = al.Galaxy(
    redshift=0.2803,
    bulge=al.lp.EllDevVaucouleurs(
        elliptical_comps=e_comps),
    mass=al.mp.EllIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.04, elliptical_comps=e_comps),
    #shear=al.mp.ExternalShear(e_comps),
    light=al.lp.EllDevVaucouleurs(
        elliptical_comps=e_comps,
        #intensity=18.04,
        #effective_radius=1.39)
        )
)

source_galaxy = al.Galaxy(
    #We know some info about the lens galaxy, but to get the source galaxy info we need to do non-linear search if we cant find it in paper
    redshift=0.982,
    bulge=al.lp.EllSersic(
        centre=(0.7, -0.05),
        elliptical_comps=e_comps,
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=2.5,
    )
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=imaging.grid)
tracer_plotter.figures_2d(image=True)

fit = al.FitImaging(imaging=imaging, tracer=tracer)

include_2d = aplt.Include2D(mask=True)

fit_imaging_plotter = aplt.FitImagingPlotter(fit=fit, include_2d=include_2d)
fit_imaging_plotter.subplot_fit_imaging()

# =============================================================================
# print("Model-Image:")
# print(fit.model_image.slim)
# print(fit.model_image.native)
# print()
# print("Residual Maps:")
# print(fit.residual_map.slim)
# print(fit.residual_map.native)
# print()
# print("Chi-Squareds Maps:")
# print(fit.chi_squared_map.slim)
# print(fit.chi_squared_map.native)
# 
# =============================================================================


#This section may be useful to double check stuff later, but pixel coordinates would have to be updated
# =============================================================================
# model_image = fit.model_image.native
# print(model_image[48:53, 48:53])
# print()
# 
# residual_map = fit.residual_map.native
# print("Residuals Central Pixels:")
# print(residual_map[48:53, 48:53])
# print()
# 
# print("Chi-Squareds Central Pixels:")
# chi_squared_map = fit.chi_squared_map.native
# print(chi_squared_map[48:53, 48:53])
# 
# =============================================================================

print("Likelihood:")
print(fit.log_likelihood)



    
    
