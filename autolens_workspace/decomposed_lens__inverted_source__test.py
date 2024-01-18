# -*- coding: utf-8 -*-
"""
Created on Sat Dec 8 01:11:14 2021

@author: kograeme
"""

"""
Using a SOURCE PARAMETRIC PIPELINE, LIGHT PIPELINE and a MASS LIGHT DARK PIPELINE 
this SLaM script  fits data, where in the final model:

 - The lens galaxy's light is a bulge+disk `EllSersic` and `EllExponential`.
 - The lens galaxy's stellar mass distribution is a bulge+disk tied to the light model above.
 - The lens galaxy's dark matter mass distribution is modeled as a `NFWSph`.
 - The source galaxy's light is a parametric `Inversion`.
"""

from pyprojroot import here
from os import path
import autolens as al
import autofit as af
import multiprocessing as mp
import slam

#Visualization
import autolens.plot as aplt
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
# %matplotlib inline

#import sys
#sys.path.insert(0, os.getcwd())

"""
__Dataset__ 

Load the image data and apply a mask
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
    shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=1.6
)

imaging = imaging.apply_mask(mask=mask)

imaging_plotter = aplt.ImagingPlotter(imaging=imaging)
imaging_plotter.subplot_imaging()

"""
__Settings AutoFit__

The settings of autofit, which controls the output paths, parallelization, database use, etc.
"""
settings_autofit = slam.SettingsAutoFit(
    path_prefix=path.join("474_output", "pipeline"),
    unique_tag=dataset_name,
    info=None,
    number_of_cores=1,
    session=None,
)

"""
__Redshifts__
"""
redshift_lens = 0.2803
redshift_source = 0.9818

"""
__HYPER SETUP__

The `SetupHyper` determines which hyper-mode features are used during the model-fit.
"""
#Note that the source hyper-mode model will be specified later, so unecessary to list
#   the hyper_galaxies_source here I believe
setup_hyper = al.SetupHyper(
    hyper_galaxies_lens=False,
    hyper_galaxies_source=False,
    hyper_image_sky=None,
    hyper_background_noise=None,
)

"""
__SOURCE PARAMETRIC PIPELINE (with lens light)__

The SOURCE PARAMETRIC PIPELINE (with lens light) uses three searches to initialize a robust model for the 
source galaxy's light, which in this example:

 - Uses a parametric `EllSersic` bulge and `EllExponential` disk with centres aligned for the lens
 galaxy's light.

 - Uses an `EllIsothermal` model for the lens's total mass distribution with an `ExternalShear`.

 __Settings__:

 - Mass Centre: Fix the mass profile centre to (0.0, 0.0) (this assumption will be relaxed in the MASS LIGHT DARK 
 PIPELINE).
"""
analysis = al.AnalysisImaging(dataset=imaging)

bulge = af.Model(al.lp.EllSersic)
disk = af.Model(al.lp.EllExponential)
bulge.centre = disk.centre

source_parametric_results = slam.source_parametric.with_lens_light(
    settings_autofit=settings_autofit,
    analysis=analysis,
    setup_hyper=setup_hyper,
    lens_bulge=bulge,
    lens_disk=disk,
    mass=af.Model(al.mp.EllIsothermal),
    shear=af.Model(al.mp.ExternalShear),
    source_bulge=af.Model(al.lp.EllSersic),
    mass_centre=(0.0, 0.0),
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
)

"""
__SOURCE INVERSION PIPELINE (with lens light)__

The SOURCE INVERSION PIPELINE (with lens light) uses four searches to initialize a robust model for the `Inversion` 
that reconstructs the source galaxy's light. It begins by fitting a `VoronoiMagnification` pixelization with `Constant` 
regularization, to set up the model and hyper images, and then:

 - Uses a `VoronoiBrightnessImage` pixelization.
 - Uses an `AdaptiveBrightness` regularization.
 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PARAMETRIC PIPELINE through to the
 SOURCE INVERSION PIPELINE.

__Settings__:

 - Positions: We update the positions and positions threshold using the previous model-fitting result (as described 
 in `chaining/examples/parametric_to_inversion.py`) to remove unphysical solutions from the `Inversion` model-fitting.
"""
settings_lens = al.SettingsLens(
    positions_threshold=source_parametric_results.last.positions_threshold_from(
        factor=3.0, minimum_threshold=0.2
    )
)

analysis = al.AnalysisImaging(
    dataset=imaging,
    hyper_dataset_result=source_parametric_results.last,
    positions=source_parametric_results.last.image_plane_multiple_image_positions,
    settings_lens=settings_lens,
)

source_inversion_results = slam.source_inversion.with_lens_light(
    settings_autofit=settings_autofit,
    analysis=analysis,
    setup_hyper=setup_hyper,
    source_parametric_results=source_parametric_results,
    pixelization=al.pix.VoronoiBrightnessImage,
    regularization=al.reg.AdaptiveBrightness,
)


"""
__LIGHT PARAMETRIC PIPELINE__

The LIGHT PARAMETRIC PIPELINE uses one search to fit a complex lens light model to a high level of accuracy, using the
lens mass model and source light model fixed to the maximum log likelihood result of the SOURCE INVERSION PIPELINE.
In this example it:

 - Uses a parametric `EllSersic` bulge and `EllExponential` disk with centres aligned for the lens galaxy's 
 light [Do not use the results of the SOURCE PARAMETRIC PIPELINE to initialize priors].

 - Uses an `EllIsothermal` model for the lens's total mass distribution [fixed from SOURCE PARAMETRIC PIPELINE].

 - Uses an `Inversion` for the source's light [priors fixed from SOURCE INVERSION PIPELINE].

 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PIPELINE through to the MASS 
 PIPELINE [fixed values].
 
"""
analysis = al.AnalysisImaging(
    dataset=imaging, hyper_dataset_result=source_inversion_results.last
)

bulge = af.Model(al.lp.EllSersic)
disk = af.Model(al.lp.EllExponential)
bulge.centre = disk.centre

light_results = slam.light_parametric.with_lens_light(
    settings_autofit=settings_autofit,
    analysis=analysis,
    setup_hyper=setup_hyper,
    source_results=source_inversion_results,
    lens_bulge=bulge,
    lens_disk=disk,
)

"""
__MASS LIGHT DARK PIPELINE (with lens light)__

The MASS LIGHT DARK PIPELINE (with lens light) uses one search to fits a complex lens mass model to a high level of 
accuracy, using the source model of the SOURCE PIPELINE and the lens light model of the LIGHT PARAMETRIC PIPELINE to 
initialize the model priors.

 - Uses a parametric `EllSersic` bulge and `EllExponential` disk with centres aligned for the lens galaxy's 
 light and its stellar mass [_ parameters: fixed from LIGHT PARAMETRIC PIPELINE].

 - The lens galaxy's dark matter mass distribution is a `SphNFW` whose centre is aligned with bulge of 
 the light and stellar mass model above [_ parameters].

 - Uses an `Inversion` for the source's light [priors fixed from SOURCE INVERSION PIPELINE].

 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PARAMETRIC PIPELINE through to the MASS 
 LIGHT DARK PIPELINE.
 
__Settings__:

 - Hyper: We may be using hyper features and therefore pass the result of the SOURCE INVERSION PIPELINE to use as the
 hyper dataset if required.

 - Positions: Update the positions and positions threshold using the previous model-fitting result to remove unphysical solutions from the `Inversion` model-fitting.
"""
settings_lens = al.SettingsLens(
    positions_threshold=source_inversion_results.last.positions_threshold_from(
        factor=3.0, minimum_threshold=0.2
    )
)

analysis = al.AnalysisImaging(
    dataset=imaging,
    hyper_dataset_result=source_inversion_results.last,
    positions=source_inversion_results.last.image_plane_multiple_image_positions,
    settings_lens=settings_lens,
)

lens_bulge = af.Model(al.lmp.EllSersic)
lens_disk = af.Model(al.lmp.EllExponential)
dark = af.Model(al.mp.SphNFW)

dark.centre = lens_bulge.centre

mass_results = slam.mass_light_dark.with_lens_light(
    settings_autofit=settings_autofit,
    analysis=analysis,
    setup_hyper=setup_hyper,
    source_results=source_inversion_results,
    light_results=light_results,
    lens_bulge=lens_bulge,
    lens_disk=lens_disk,
    lens_envelope=None,
    dark=dark,
)

