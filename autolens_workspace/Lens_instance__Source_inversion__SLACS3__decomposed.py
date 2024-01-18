# -*- coding: utf-8 -*-
"""
@author: kograeme
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
dataset_name = "slacs1430+4105"
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
    radius=2.9
)

imaging = imaging.apply_mask(mask=mask)

#Approximate image positions to by searching the arcs that sweep through these positions
positions = al.Grid2DIrregular.from_json(
    file_path=path.join(dataset_path, "positions.json")
)

#Setting the threshold for positions
settings_lens = al.SettingsLens(positions_threshold=1.0)

#Plot images with position markers
visuals_2d = aplt.Visuals2D(positions=positions)
imaging_plotter = aplt.ImagingPlotter(imaging=imaging, visuals_2d=visuals_2d)
imaging_plotter.subplot_imaging()

"""
__Paths__

The path the results of decomposed model tests are output:
"""

path_prefix=path.join("474_output", "model_test", "SLACS3")

"""
__Redshifts__
The redshifts of the lens and source galaxies, which are used to perform unit converions of the model and data (e.g. 
from arc-seconds to kiloparsecs, masses to solar masses, etc.).
"""

redshift_lens = 0.2850
redshift_source =  0.5753
"""
__Model + Search + Analysis + Model-Fit__

In a non-linear search we fit a lens model where:

 - The lens galaxy's stellar light and mass is a parametric `EllSersic` bulge and `EllExponential` disk.
 
 - The lens galaxy has a dark mass profile 'SphNFW'

 - The source galaxy is fitted by inversion with adaptive pixellization and regularization.

The number of free parameters and therefore the dimensionality of non-linear parameter space is N= ____.
"""

setup_hyper = al.SetupHyper(
    hyper_galaxies_lens=False,
    hyper_galaxies_source=False,
    hyper_image_sky=al.hyper_data.HyperImageSky,
    hyper_background_noise=None,
)

#Passing priors here to make search go faster
sers_ecomps = al.convert.elliptical_comps_from(0.922, 92)
exp_ecomps = al.convert.elliptical_comps_from(0.683, 70)

print(f"Sersic set ecomps: {sers_ecomps}")
print(f"Exponential set ecomps: {exp_ecomps}")

#Setting the profiles for the lens model
sers = al.lmp.EllSersic
exp = al.lmp.EllExponential
dark = al.mp.SphNFW

sers.centre_0 = 0.028
sers.centre_1 = 0.0306
sers.intensity = 0.166
sers.sersic_index = 2.86
sers.effective_radius = 0.587
sers.elliptical_comps = sers_ecomps
sers.mass_to_light_ratio = 3.37

exp.centre_0 = -0.013 
exp.centre_1 = -0.002
exp.intensity = 0.016
exp.effective_radius = 2.8
exp.elliptical_comps=exp_ecomps
exp.mass_to_light_ratio = 7.71
#Exponential profile sersic index is 1 

#sers.centre = exp.centre -> Not for SLACS3

dark.centre = (0,0) #maybe change dark centre prior later. Can't find anything in paper that says dark center not (0,0)
scale_radius = 10*2.55 #Approximated using the same assumption as the Nightingale paper. Maybe try different values
dark.scale_radius = scale_radius 
dark.kappa_s=0.054

#Lens has everything but shear
lens_search1_model = af.Model(
    al.Galaxy,
    redshift=redshift_lens, 
    light_1=sers, 
    light_2=exp, 
    dark=dark,
    shear=al.mp.ExternalShear
)

#Use voronoi tesselation for source inversion
source_search1_model=af.Model(
    al.Galaxy,
    redshift=redshift_source,
    pixelization=al.pix.VoronoiMagnification,
    regularization=al.reg.Constant
)
#If we want to use hyper-mode later, then using pix.VoronoiMagnification and
#   reg.Constant will provide a far more accurate hyper-image to use than if
#   we were to apply hyper-mode on a parametric source, which would likely
#   end in failure. 


model = af.Collection(
    galaxies=af.Collection(
        lens=lens_search1_model,
        source=source_search1_model
    )
)

search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="[01]decomp__instancelens__inverse_source__nonhyper",
    unique_tag=dataset_name,
    nlive=50,
    walks=7,
    dlogz=1
)

analysis = al.AnalysisImaging(
    dataset=imaging, positions=positions, settings_lens=settings_lens
)

result_1 = search.fit(model=model, analysis=analysis)

result_1 = extensions.hyper_fit(
    setup_hyper=setup_hyper,
    result=result_1,
    analysis=analysis
)

print(f"result_1 max likelihood lens galaxy: {result_1.max_log_likelihood_instance.galaxies.lens}")
print(f"result_1 max likelihood lens galaxy: {result_1.max_log_likelihood_instance.galaxies.source}")

dynesty_plotter = aplt.DynestyPlotter(samples=result_1.samples)
dynesty_plotter.cornerplot()

tracer_plotter = aplt.TracerPlotter(tracer=result_1.max_log_likelihood_tracer, grid=mask.masked_grid)
tracer_plotter.subplot_tracer()

fit_imaging_plotter = aplt.FitImagingPlotter(fit=result_1.max_log_likelihood_fit)
fit_imaging_plotter.subplot_fit_imaging()
fit_imaging_plotter.subplot_of_planes(plane_index=1)

#Plot source reconstruction for search 1
mat_plot_2d = aplt.MatPlot2D(cmap=aplt.Cmap(cmap='gray'))

inversion_plotter = aplt.InversionPlotter(
    inversion=result_1.max_log_likelihood_fit.inversion, mat_plot_2d=mat_plot_2d
)
inversion_plotter.figures_2d_of_mapper(mapper_index=0, reconstruction=True)


# #Hyper-mode start for search 2
#hyper_image = result_1.max_log_likelihood.model_image.binned.slim

source_search2_adaptive = af.Model(
    al.Galaxy,
    redshift=redshift_source,
    pixelization=al.pix.VoronoiBrightnessImage(pixels=400, weight_floor=0.0, weight_power=1),
    regularization=al.reg.AdaptiveBrightness,
    hyper_galaxy_image=result_1.instance.galaxies.source.hyper_galaxy
)

model = af.Collection(
    galaxies=af.Collection(
        lens=result_1.instance.galaxies.lens,
        source=source_search2_adaptive
    )
)

search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="[02]decomp__instancelens__inverse_source__hyper",
    unique_tag=dataset_name,
    nlive=120,
    walks=8,
    dlogz=0.8
)

analysis = al.AnalysisImaging(
    dataset=imaging, positions=positions, settings_lens=settings_lens
)

result_2 = search.fit(model=model, analysis=analysis)



