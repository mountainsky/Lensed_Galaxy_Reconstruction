"""
This script aims to test the decomposed lens model of the Nightingale paper 
    https://arxiv.org/pdf/1901.07801.pdf. It will start a non-linear search near
    the lens parameters that were given in the paper yet still leave room for adjustments.
    It will also attempt to reconstruct the source based on adaptive pixellization
    and its model fit with the given lens parameters. 
"""

from pyprojroot import here
from os import path
import autolens as al
import autofit as af
import multiprocessing as mp

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

#Approximate image positions to search through with positions file
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

#Passing priors here to make search go faster
bulge_ecomps = al.convert.elliptical_comps_from(0.922, 92)
disk_ecomps = al.convert.elliptical_comps_from(0.683, 70)

print(f"Bulge prior ecomps: {bulge_ecomps}")
print(f"Disk prior ecomps: {disk_ecomps}")

#Setting the profiles for the lens model
sers = af.Model(al.lmp.EllSersic)
exp = af.Model(al.lmp.EllExponential)
dark = af.Model(al.mp.SphNFW)

sers.centre_0 = af.GaussianPrior(mean=0.028, sigma=0.001)
sers.centre_1 = af.GaussianPrior(mean=0.0306, sigma=0.001)
sers.intensity = af.GaussianPrior(mean=0.166, sigma=0.02)
sers.sersic_index = af.GaussianPrior(mean=2.86, sigma=0.15)
sers.effective_radius = af.GaussianPrior(mean=0.587, sigma=0.5)
sers.elliptical_comps.elliptical_comps_0 = af.GaussianPrior(mean=bulge_ecomps[0], sigma=0.05)
sers.elliptical_comps.elliptical_comps_1 = af.GaussianPrior(mean=bulge_ecomps[1], sigma=0.05)
sers.mass_to_light_ratio = af.GaussianPrior(mean=3.37, sigma=0.7)

exp.centre_0 = af.GaussianPrior(mean=-0.013, sigma=0.02) 
exp.centre_1 = af.GaussianPrior(mean=-0.002, sigma=0.02)
exp.intensity = af.GaussianPrior(mean=0.016, sigma=0.01)
exp.effective_radius = af.GaussianPrior(mean=2.8, sigma=0.2)
exp.elliptical_comps.elliptical_comps_0= af.GaussianPrior(mean=disk_ecomps[0], sigma=0.05)
exp.elliptical_comps.elliptical_comps_1 = af.GaussianPrior(mean=disk_ecomps[1], sigma=0.05)
exp.mass_to_light_ratio = af.GaussianPrior(mean=7.71, sigma=0.4)
#Exponential profile sersic index is 1 

#sers.centre = exp.centre -> Not for SLACS3

dark.centre_0=af.UniformPrior(lower_limit=-0.1, upper_limit=0.1) #maybe change dark centre prior later
dark.centre_1=af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
scale_radius = 10*2.8 #Approximated using the same assumption as the Nightingale paper
dark.scale_radius = scale_radius 
dark.kappa_s=af.GaussianPrior(mean=0.054,sigma=0.02)


#Use voronoi tesselation for source inversion
source_search1_model=af.Model(
    al.Galaxy,
    redshift=redshift_source,
    pixelization=al.pix.VoronoiMagnification(shape=(30,30)),
    regularization=al.reg.Constant(coefficient=1.5)
)
#If we want to use hyper-mode later, then using pix.VoronoiMagnification and
#   reg.Constant will provide a far more accurate hyper-image to use than if
#   we were to apply hyper-mode on a parametric source, which would likely
#   end in failure. 


#Will have to look more into it, but maybe we should treat lens as a hyper-galaxy too?

#Shear not given in paper

model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(al.Galaxy, redshift=redshift_lens, light_1=sers, light_2=exp, dark=dark, shear=al.mp.ExternalShear),
        source=source_search1_model
    )
)

search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="decomp__gausspriorlens__inverse_source__nonhyper",
    unique_tag=dataset_name,
    nlive=300,
    walks=9
)

analysis = al.AnalysisImaging(
    dataset=imaging, positions=positions, settings_lens=settings_lens
)

result = search.fit(model=model, analysis=analysis)

print(result.max_log_likelihood_instance.galaxies.lens)
print(result.max_log_likelihood_instance.galaxies.source)

dynesty_plotter = aplt.DynestyPlotter(samples=result.samples)
dynesty_plotter.cornerplot()

tracer_plotter = aplt.TracerPlotter(tracer=result.max_log_likelihood_tracer, grid=mask.masked_grid)
tracer_plotter.subplot_tracer()

fit_imaging_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_imaging_plotter.subplot_fit_imaging()
