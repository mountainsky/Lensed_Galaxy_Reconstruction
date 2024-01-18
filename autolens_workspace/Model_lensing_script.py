from pyprojroot import here
from os import path
import autolens as al
import autolens.plot as aplt
from astropy.io import fits
import autofit as af
import numpy as np
import multiprocessing as mp

#Visualization
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)

workspace_path = str(here())
dataset_path = path.join(workspace_path,"474_project", "474_data","nightingale","slacs_new","slacs0252+0039")

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "F814W_image.fits"),
    noise_map_path=path.join(dataset_path, "F814W_noise_map.fits"),
    psf_path=path.join(dataset_path, "F814W_psf.fits"),
    pixel_scales=0.05,
    image_hdu=0, 
    noise_map_hdu=0,
    psf_hdu=0
)

imaging_plotter = aplt.ImagingPlotter(imaging=imaging)
imaging_plotter.subplot_imaging()

#Smaller mask radius allows for better computation time, but maybe worse results
mask = al.Mask2D.circular(
    shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=2)

visuals_2d = aplt.Visuals2D(mask=mask)
imaging_plotter = aplt.ImagingPlotter(imaging=imaging, visuals_2d=visuals_2d)
imaging_plotter.figures_2d(image=True)

imaging = imaging.apply_mask(mask=mask)

include_2d = aplt.Include2D(mask=True)
imaging_plotter = aplt.ImagingPlotter(imaging=imaging, include_2d=include_2d)
imaging_plotter.figures_2d(image=True)

#Now trying non-linear search methods
#Should try various methods to decrease computation time. If running on linux, could also do multiprocessing

e_comps = al.convert.elliptical_comps_from(axis_ratio=0.93, angle=106.2)

lens = af.Model(
    al.Galaxy,
    redshift=0.2803,
    light=al.lp.EllSersic,
    mass=al.mp.EllPowerLaw,
    shear=al.mp.ExternalShear
)

source = af.Model(al.Galaxy, redshift=0.9818, light1=al.lp.EllSersic)

#The following lines are assertions to try to make the non-linear search faster
#lens.mass.centre_0 = af.UniformPrior(lower_limit=-0.3, upper_limit=0.3)
#lens.mass.centre_1 = af.UniformPrior(lower_limit=-0.3, upper_limit=0.3)
#lens.mass.einstein_radius = af.GaussianPrior(mean=1.04, sigma=0.2, lower_limit=0.0, upper_limit=np.inf)
#lens.mass.slope = af.GaussianPrior(mean=1.57, sigma=0.4, lower_limit=0.0, upper_limit=np.inf)

# =============================================================================
# print(f"Parameter searching around ellipticity components: {e_comps}")
# lens.mass.elliptical_comps.elliptical_comps_0 = af.GaussianPrior(
#     mean=e_comps[0], sigma=0.1, lower_limit=-1.0, upper_limit=1.0)
# lens.mass.elliptical_comps.elliptical_comps_1 = af.GaussianPrior(
#     mean=e_comps[1], sigma=0.1, lower_limit=-1.0, upper_limit=1.0)
# 
# lens.light.elliptical_comps.elliptical_comps_0 = af.GaussianPrior(
#     mean=e_comps[0], sigma=0.1, lower_limit=-1.0, upper_limit=1.0)
# lens.light.elliptical_comps.elliptical_comps_1 = af.GaussianPrior(
#     mean=e_comps[1], sigma=0.1, lower_limit=-1.0, upper_limit=1.0) 
# =============================================================================

lens.mass.elliptical_comps = lens.light.elliptical_comps = e_comps
lens.mass.centre = lens.light.centre = (0.0,0.0)
lens.mass.einstein_radius = 1.04
lens.light.effective_radius= 1.39
lens.light.sersic_index = af.GaussianPrior(mean=4, sigma=0.2, lower_limit=1, upper_limit=np.inf)

#Not sure if the source effective radius is based on the image plane or the source plane
source.light1.effective_radius = af.UniformPrior(lower_limit=0.01, upper_limit=0.3)
    
galaxies = af.Collection(lens=lens, source=source)
model = af.Collection(galaxies=galaxies)

#Constrain the model to fit these source image arc positions
positions = al.Grid2DIrregular(grid = [(-0.925, 0.025), (-0.875, 0.225), (1.075, 0.275), (0.9750000000000001, -0.5750000000000001)])
settings_lens = al.SettingsLens(positions_threshold=0.7)
#Note that positions and settings_lens are arguments for the AnalysisImaging object. 

analysis = al.AnalysisImaging(dataset=imaging, positions=positions, settings_lens=settings_lens)

#Would be nice to get multiprocessing working on windows for the dynesty search 
search = af.DynestyStatic(name="search_manual",
    path_prefix=path.join("474_output", "final"),
    unique_tag="slacs0252+0039",
    nlive=150,
    walks=7,
    dlogz=0.8,
    number_of_cores=1)

result = search.fit(model=model, analysis=analysis)

print(result.max_log_likelihood_instance.galaxies.lens)
print(result.max_log_likelihood_instance.galaxies.source)

dynesty_plotter = aplt.DynestyPlotter(samples=result.samples)
dynesty_plotter.cornerplot()

tracer_plotter = aplt.TracerPlotter(tracer=result.max_log_likelihood_tracer, grid=mask.masked_grid)
tracer_plotter.subplot_tracer()

fit_imaging_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_imaging_plotter.subplot_fit_imaging()