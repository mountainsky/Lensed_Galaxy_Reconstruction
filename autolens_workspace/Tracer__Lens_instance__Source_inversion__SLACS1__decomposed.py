# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 15:35:06 2021

@author: kogra
"""

"""
This script aims to test the decomposed lens model of the Nightingale paper 
    https://arxiv.org/pdf/1901.07801.pdf. It will start a non-linear search using
    the lens parameters that were given in the paper and use that lens model to
    fit a VoronoiMagnification pixelization. Then it will be passed to another
    non-linear search where hyper-mode is utilized
"""

from pyprojroot import here
from os import path
import autolens as al
import autofit as af
import multiprocessing as mp
import extensions

#Visualization
import autolens.plot as aplt
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
#plt.style.use(astropy_mpl_style)


"""
__Dataset__ 

Load the `Imaging` data, define the `Mask2D` and plot them.
"""

workspace_path = str(here())
dataset_name = "slacs0252+0039"
dataset_path = path.join(workspace_path,"474_project", "474_data","nightingale","slacs_new", dataset_name)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "F814W_image.fits"),
    noise_map_path=path.join(dataset_path, "F814W_noise_map.fits"),
    psf_path=path.join(dataset_path, "F814W_psf.fits"),
    pixel_scales=0.05,
    image_hdu=0, 
    noise_map_hdu=0,
    psf_hdu=0
)

mask = al.Mask2D.circular(
    shape_native=imaging.shape_native,
    pixel_scales=imaging.pixel_scales,
    radius=1.6
)

imaging = imaging.apply_mask(mask=mask)


"""
__Paths__

The path the results of decomposed model tests are output:
"""

path_prefix=path.join("474_output", "model_test", "SLACS1")

"""
__Redshifts__
The redshifts of the lens and source galaxies, which are used to perform unit converions of the model and data (e.g. 
from arc-seconds to kiloparsecs, masses to solar masses, etc.).
"""

redshift_lens = 0.28
redshift_source =  0.982


#Setting the profiles for the lens model
sers = al.mp.EllSersic
exp = al.mp.EllExponential
dark = al.mp.SphNFW


sers_ecomps = al.convert.elliptical_comps_from(0.889, 38)
exp_ecomps = al.convert.elliptical_comps_from(0.824, 106.2)

print(f"Sersic set ecomps: {sers_ecomps}")
print(f"Exponential set ecomps: {exp_ecomps}")

#Setting the profiles for the lens model
sers = al.lmp.EllSersic
exp = al.lmp.EllExponential
dark = al.mp.SphNFW

sers.intensity = 0.448
sers.sersic_index = 2.36
sers.effective_radius = 0.141
sers.elliptical_comps = sers_ecomps

exp.intensity = 0.075
exp.effective_radius = 0.811
exp.elliptical_comps=exp_ecomps
#Exponential profile sersic index is 1 

#sers.mass_to_light_ratio = exp.mass_to_light_ratio = 5.26
sers.centre = exp.centre = dark.centre = (0,0)

scale_radius = 10*1.39 #Approximated using the same assumption as the Nightingale paper. Maybe try different values
dark.scale_radius = scale_radius 
dark.kappa_s= 0.042


#Don't have the shear values -> Perhaps can omit
lens_galaxy = al.Galaxy(
    redshift=redshift_lens, 
    bulge=sers, 
    disk=exp,
    dark=dark
)


source_galaxy_magnification = al.Galaxy(
    redshift=redshift_source,
    pixelization=al.pix.VoronoiMagnification(shape=(25,25)),
    regularization=al.reg.Constant(coefficient=1.5),
)


#This code is just for tracer fitting direct models
tracer = al.Tracer.from_galaxies(galaxies=[source_galaxy_magnification, lens_galaxy])

fit = al.FitImaging(imaging=imaging, tracer=tracer)

print(f"Fit of Voronoi pixelization and constant regularization gives evidence of {fit.log_evidence}")

include_2d = aplt.Include2D(mask=True, mapper_data_pixelization_grid=True, mapper_source_pixelization_grid=True)

fit_imaging_plotter = aplt.FitImagingPlotter(fit=fit, include_2d=include_2d)
fit_imaging_plotter.subplot_fit_imaging()
fit_imaging_plotter.subplot_of_planes(plane_index=1)

mat_plot_2d = aplt.MatPlot2D(cmap=aplt.Cmap(cmap='gray'))

inversion_plotter = aplt.InversionPlotter(
    inversion=fit.inversion, mat_plot_2d=mat_plot_2d
)
inversion_plotter.figures_2d_of_mapper(mapper_index=0, reconstruction=True)

cosmology = tracer.cosmology
print("Source-plane kpc-per-arcsec:")
print(al.util.cosmology.kpc_per_arcsec_from(redshift=redshift_source, cosmology=cosmology))

print("Lens-plane kpc-per-arcsec:")
print(al.util.cosmology.kpc_per_arcsec_from(redshift=redshift_lens, cosmology=cosmology))


#Hyper-mode start for fit
hyper_image = fit.model_image.binned.slim

source_galaxy_brightness = al.Galaxy(
    redshift=redshift_source,
    pixelization=al.pix.VoronoiBrightnessImage(
        pixels = 350, weight_floor=0.0, weight_power=1),
    regularization=al.reg.AdaptiveBrightness(
        inner_coefficient=0.01, outer_coefficient=0.2,signal_scale=2),
    hyper_galaxy_image=hyper_image,
)

tracer2 = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy_brightness])

fit2 = al.FitImaging(imaging=imaging, tracer=tracer2)

print(f"Fit of adaptive pixelization gives evidence of {fit2.log_evidence}")

fit_imaging_plotter = aplt.FitImagingPlotter(fit=fit2, include_2d=include_2d)
fit_imaging_plotter.subplot_fit_imaging()
fit_imaging_plotter.subplot_of_planes(plane_index=1)

inversion_plotter = aplt.InversionPlotter(
    inversion=fit2.inversion, mat_plot_2d=mat_plot_2d
)
inversion_plotter.figures_2d_of_mapper(mapper_index=0, reconstruction=True)

