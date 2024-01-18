"""
Pipelines: Light Parametric + Mass Light Dark + Source Inversion
================================================================

By chaining together five searches this script fits `Imaging` dataset of a 'galaxy-scale' strong lens, where in the
final model:

 - The lens galaxy's light is a parametric bulge+disk `EllSersic` and `EllExponential`.
 - The lens galaxy's stellar mass distribution is a bulge+disk tied to the light model above.
 - The source galaxy is modeled using an `Inversion`.
"""

'''
Below script uses SLaM for SLACS 3
The final DECOMPOSED model:
 - Lens galaxy's light is a bulge+disk 'EllSersic' and 'EllExponential'
 - Lens galaxy's stellar mass distribution is a bulge+disk tied to the light model above.
 - Lens galaxy's dark matter mass distribution is modeled as a `SphNFW`.
 - Source galaxy's light is a parametric 'Inversion'
The following are the inputs from Nightingale:
 - redshift_lens: 0.2850
 - redshift_source: 0.5753
 - mask inner radius: 0.4
 - mask outer radius: 2.9
 - effective radius: 2.55
 - axis ratio: 0.68
 - Position angle: 111.7 (For SIE, See Table 2 of Paper)
 - einstein radius: 1.52 (ONLY for Total mass model)
'''

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
# %matplotlib inline

#import sys
#sys.path.insert(0, os.getcwd())

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

#For this preliminary search, I think we can use mask out the source light
mask = al.Mask2D.circular(
    shape_native=imaging.shape_native,
    pixel_scales=imaging.pixel_scales,
    radius=0.4 
)

imaging = imaging.apply_mask(mask=mask)

"""
__Paths__

The path the results of all chained searches are output:
"""
path_prefix=path.join("474_output", "pipeline", "decomposed_mass")

"""
__Redshifts__

The redshifts of the lens and source galaxies, which are used to perform unit converions of the model and data (e.g. 
from arc-seconds to kiloparsecs, masses to solar masses, etc.).

In this analysis, they are used to explicitly set the `mass_at_200` of the elliptical NFW dark matter profile, which is
a model parameter that is fitted for.
"""

redshift_lens = 0.2850
redshift_source =  0.5753

"""
__Model + Search + Analysis + Model-Fit (Search 1)__

In search 1 we fit a lens model where:

 - The lens galaxy's light is a parametric `EllSersic` bulge and `EllExponential` disk, the centres of 
 which are aligned [11 parameters].

 - The lens galaxy's mass and source galaxy are omitted.

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=11.
"""

#Passing priors here to make search go faster
#bulge_ecomps = al.convert.elliptical_comps_from(0.922, 92)
#disk_ecomps = al.convert.elliptical_comps_from(0.683, 70)
#print(f"Bulge prior ecomps: {bulge_ecomps}")
#print(f"Disk prior ecomps: {disk_ecomps}")

bulge = af.Model(al.lp.EllSersic)
disk = af.Model(al.lp.EllExponential)

# =============================================================================
# bulge.centre_0 = af.GaussianPrior(mean=0.028, sigma=0.001)
# bulge.centre_1 = af.GaussianPrior(mean=0.0306, sigma=0.001)
# bulge.intensity = af.GaussianPrior(mean=0.166, sigma=0.02)
# bulge.sersic_index = af.GaussianPrior(mean=2.86, sigma=0.15)
# bulge.effective_radius = af.GaussianPrior(mean=0.587, sigma=0.5)
# bulge.elliptical_comps.elliptical_comps_0 = af.GaussianPrior(mean=bulge_ecomps[0], sigma=0.1)
# bulge.elliptical_comps.elliptical_comps_1 = af.GaussianPrior(mean=bulge_ecomps[1], sigma=0.1)
# 
# disk.centre_0 = af.GaussianPrior(mean=-0.013, sigma=0.02) 
# disk.centre_1 = af.GaussianPrior(mean=-0.002, sigma=0.02)
# disk.intensity = af.GaussianPrior(mean=0.016, sigma=0.01)
# disk.effective_radius = af.GaussianPrior(mean=2.8, sigma=0.2)
# disk.elliptical_comps.elliptical_comps_0= af.GaussianPrior(mean=disk_ecomps[0], sigma=0.1)
# disk.elliptical_comps.elliptical_comps_1 = af.GaussianPrior(mean=disk_ecomps[1], sigma=0.1)
# #Disk sersic index is 1 for exponential profile
# 
# =============================================================================
#Will ignore mass-to-light ratio priors for now -> also maybe should keep this purely about light profile for now?

#bulge.centre = disk.centre -> Not for SLACS3

model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(al.Galaxy, redshift=redshift_lens, bulge=bulge, disk=disk)
    )
)

search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="search[1]_light[parametric]_onlylens",
    unique_tag=dataset_name,
    nlive=500,
    walks=9
)


analysis = al.AnalysisImaging(dataset=imaging)

result_1 = search.fit(model=model, analysis=analysis)

"""
__Model + Search + Analysis + Model-Fit (Search 2)__

We use the results of search 1 to create the lens model fitted in search 2, where:

 - The lens galaxy's light and stellar mass is an `EllSersic` bulge and `EllExponential` disk [Parameters 
 fixed to results of search 1].

 - The lens galaxy's dark matter mass distribution is a `SphNFW` whose centre is aligned with the 
 `EllSersic` bulge and stellar mass model above [3 parameters].

 - The lens mass model also includes an `ExternalShear` [2 parameters].

 - The source galaxy's light is a parametric `EllSersic` [7 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=12.

NOTES:

 - By using the fixed `bulge` and `disk` model from the result of search 1, we are assuming this is a sufficiently 
 accurate fit to the lens's light that it can reliably represent the stellar mass.
"""

#Use new mask that includes source light now
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


bulge = result_1.instance.galaxies.lens.bulge
disk = result_1.instance.galaxies.lens.disk

dark = af.Model(al.mp.SphNFW)
dark.centre = bulge.centre
dark.mass_at_200 = af.LogUniformPrior(lower_limit=1e8, upper_limit=1e15)
dark.redshift_object = redshift_lens
dark.redshift_source = redshift_source

source_bulge=al.lp.EllSersic
source_bulge.centre_0 = af.UniformPrior(lower_limit=-0.3, upper_limit=0.3)
source_bulge.centre_1 = af.UniformPrior(lower_limit=-0.3, upper_limit=0.3)

model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            redshift=redshift_lens,
            bulge=bulge,
            disk=disk,
            dark=af.Model(al.mp.SphNFW),
            shear=al.mp.ExternalShear,
        ),
        source=af.Model(al.Galaxy, redshift=redshift_source, bulge=source_bulge),
    )
)

search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="search[2]_light[fixed]_mass[light_dark]_source[parametric]",
    unique_tag=dataset_name,
    nlive=144,
    walks=7,
    dlogz=0.8
)


"""
Positions are chosen from positions.json file. The positions are plotted
"""

positions = al.Grid2DIrregular.from_json(
    file_path=path.join(dataset_path, "positions.json")
)

visuals_2d = aplt.Visuals2D(positions=positions)

imaging_plotter = aplt.ImagingPlotter(imaging=imaging, visuals_2d=visuals_2d)
imaging_plotter.subplot_imaging()

#Setting the threshold for positions
settings_lens = al.SettingsLens(positions_threshold=1.0)

analysis = al.AnalysisImaging(
    dataset=imaging, positions=positions, settings_lens=settings_lens
)


result_2 = search.fit(model=model, analysis=analysis)

"""
__Model + Search + Analysis + Model-Fit (Search 3)__

We use the results of searches 1 and 2 to create the lens model fitted in search 3, where:

 - The lens galaxy's light and stellar mass is a parametric `EllSersic` bulge and `EllExponential` disk 
 [8 parameters: priors initialized from search 1].

 - The lens galaxy's dark matter mass distribution is a `SphNFW` whose centre is aligned with the 
 `EllSersic` bulge and stellar mass model above [3 parameters: priors initialized from search 2].

 - The lens mass model also includes an `ExternalShear` [2 parameters: priors initialized from search 2].

 - The source galaxy's light is a parametric `EllSersic` [7 parameters: priors initialized from search 2].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=22.

Notes:

 - This search attempts to address any issues there may have been with the bulge's stellar mass model.
"""

bulge = bulge #Instance from result 1, was getting errors for result_1.model. Maybe should set model/instance parameters directly
disk = disk

#bulge = result_1.model.galaxies.lens.bulge
#disk = result_1.model.galaxies.lens.disk

dark = result_2.model.galaxies.lens.dark
dark.centre = bulge.centre

model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            redshift=redshift_lens,
            bulge=bulge,
            disk=disk,
            dark=dark,
            shear=result_2.model.galaxies.lens.shear,
        ),
        source=af.Model(
            al.Galaxy, redshift=redshift_source, bulge=result_2.model.galaxies.source.bulge
        ),
    )
)

search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="search[3]_light[parametric]_mass[light_dark]_source[parametric]",
    unique_tag=dataset_name,
    nlive=200,
    walks=7,
    dlogz=0.8
)

analysis = al.AnalysisImaging(dataset=imaging, positions=positions,settings_lens=settings_lens)

result_3 = search.fit(model=model, analysis=analysis)

"""
__Model + Search + Analysis + Model-Fit (Search 4)__

We use the results of searches 3 to create the lens model fitted in search 4, where:

 - The lens galaxy's light and stellar mass is an `EllSersic` bulge and `EllExponential` 
 disk [Parameters fixed to results of search 3].

 - The lens galaxy's dark matter mass distribution is a `SphNFW [Parameters fixed to results of 
 search 3].

 - The lens mass model also includes an `ExternalShear` [Parameters fixed to results of search 3].

 - The source-galaxy's light uses a `VoronoiMagnification` pixelization [2 parameters].

 - This pixelization is regularized using a `Constant` scheme [1 parameter]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=3.

NOTES:

 - This search allows us to very efficiently set up the resolution of the pixelization and regularization coefficient 
 of the regularization scheme, before using these models to refit the lens mass model.
"""
model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            redshift=redshift_lens,
            bulge=result_3.instance.galaxies.lens.bulge,
            disk=result_3.instance.galaxies.lens.disk,
            dark=result_3.instance.galaxies.lens.dark,
            shear=result_3.instance.galaxies.lens.shear,
        ),
        source=af.Model(
            al.Galaxy,
            redshift=redshift_source,
            pixelization=al.pix.VoronoiMagnification,
            regularization=al.reg.Constant,
        ),
    )
)

search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="search[4]_light[fixed]_mass[fixed]_source[inversion_initialization]",
    unique_tag=dataset_name,
    nlive=20,
    walks=9,
    dlogz=0.8
)

analysis = al.AnalysisImaging(dataset=imaging,positions=positions,settings_lens=settings_lens)

result_4 = search.fit(model=model, analysis=analysis)

"""
__Model +  Search (Search 5)__

We use the results of searches 3 and 4 to create the lens model fitted in search 5, where:

 - The lens galaxy's light and stellar mass is an `EllSersic` bulge and `EllExponential` 
 disk [priors initialized from search 3].

 - The lens galaxy's dark matter mass distribution is a `SphNFW` [priors initialized 
 from search 3].

The lens mass model also includes an `ExternalShear` [2 parameters: priors initialized from search 3].

 - The source-galaxy's light uses a `VoronoiMagnification` pixelization [parameters fixed to results of search 4].

 - This pixelization is regularized using a `Constant` scheme [parameters fixed to results of search 4]. 
"""
model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            redshift=redshift_lens,
            bulge=result_3.model.galaxies.lens.bulge,
            disk=result_3.model.galaxies.lens.disk,
            dark=result_3.model.galaxies.lens.dark,
            shear=result_3.model.galaxies.lens.shear,
        ),
        source=af.Model(
            al.Galaxy,
            redshift=redshift_source,
            pixelization=result_4.instance.galaxies.source.pixelization,
            regularization=result_4.instance.galaxies.source.regularization,
        ),
    )
)

search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="search[5]_light[parametric]_mass[light_dark]_source[inversion]",
    unique_tag=dataset_name,
    nlive=150,
    walks=8,
    dlogz=0.2
)

"""
__Positions + Analysis + Model-Fit (Search 5)__

We update the positions and positions threshold using the previous model-fitting result (as described 
 in `chaining/examples/parametric_to_inversion.py`) to remove unphysical solutions from the `Inversion` model-fitting.
"""
settings_lens = al.SettingsLens(
    positions_threshold=result_4.positions_threshold_from(
        factor=3.0, minimum_threshold=0.2
    )
)

analysis = al.AnalysisImaging(
    dataset=imaging,
    settings_lens=settings_lens,
    positions=result_4.image_plane_multiple_image_positions,
)

result_5 = search.fit(model=model, analysis=analysis)

"""
Finish.
"""
