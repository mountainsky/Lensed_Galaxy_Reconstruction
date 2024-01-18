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

imaging_plotter = aplt.ImagingPlotter(imaging=imaging)
imaging_plotter.subplot_imaging()

'''
AUTOFIT
output paths, parellelization, database use, etc
-------------------
   Parameters
-------------------
path_prefix
    The prefix of folders between the output path and the search folders.
unique_tag
    The unique tag for this model-fit, which will be given a unique entry in the sqlite database and also acts as
    the folder after the path prefix and before the search name. This is typically the name of the dataset.
info : dict
    Optional dictionary containing information about the model-fit that is stored in the database and can be
    loaded by the aggregator after the model-fit is complete.
number_of_cores
    The number of CPU cores used to parallelize the model-fit. This is used internally in a non-linear search
    for most model fits, but is done on a per-fit basis for grid based searches (e.g. sensitivity mapping).
session
    The SQLite database session which is active means results are directly wrtten to the SQLite database
    at the end of a fit and loaded from the database at the start
'''

settings_autofit = slam.SettingsAutoFit(
    path_prefix=path.join("474_output", "pipeline", "decomposed_mass"),
    unique_tag=dataset_name,
    info=None,
    number_of_cores=1,
    session=None,
)

'''
REDSHIFT
'''

redshift_lens = 0.2850
redshift_source =  0.5753

'''
HYPER MODE
Taken from https://pyautolens.readthedocs.io/en/latest/api/generated/autolens.SetupHyper.html?highlight=l.SetupHyper#autolens.SetupHyper
The hyper setup of a pipeline, which controls how hyper-features in PyAutoLens template pipelines run,
for example controlling whether hyper galaxies are used to scale the noise and the non-linear searches used in these searchs.
----------------
Parameters
---------------
hyper_galaxies
    Determines if hyper-galaxy functionality is used to scale the noise-map of the dataset throughout the fitting.
hyper_image_sky
    Determines if hyper-galaxy functionality is used include the image’s background sky component in the model.
hyper_background_noise
    Determines if hyper-galaxy functionality is used include the noise-map’s background component in the model.
hyper_fixed_after_source
    If True, the hyper parameters are fixed and not updated after a desnated pipeline in the analysis.
    For the SLaM pipelines this is after the SourcePipeline.
    This allow Bayesian model comparison to be performed objected between later searchs in a pipeline.
search_inversion_cls
    The non-linear search used by every hyper model-fit search
search_inversion_dict
    The dictionary of search options for the hyper model-fit searches.
'''

setup_hyper = al.SetupHyper(
    hyper_galaxies_lens=False,
    hyper_galaxies_source=False,
    hyper_image_sky=None,
    hyper_background_noise=None,
)

'''
SOURCE PARAMETRIC PIPELINE
- Parametric `EllSersic` bulge and `EllExponential` disk with centres aligned for the lens galaxy's light.
- 'EllIsothermal' model for the len's total mass distribution with an 'ExternalShear'
- Mass Centre: Fix the mass profile centre to (0.0, 0.0) (this assumption will be relaxed in the MASS LIGHT DARK PIPELINE)
Positions are chosen from positions.json file. The positions are plotted
The SlaM SOURCE PARAMETRIC PIPELINE for fitting imaging data with a lens light component.
The pipeline has three searches
In search 1 of the SOURCE PARAMETRIC PIPELINE we fit a lens model where:
     - The lens galaxy light is modeled using parametric bulge + disk + envelope [no prior initialization].
     - The lens's mass and source galaxy are omitted from the fit.
This search aims to produce a somewhat accurate lens light subtracted image for the next search which fits the
the lens mass model and source model.
In search 2 of the SOURCE PARAMETRIC PIPELINE we fit a lens model where:
     - The lens galaxy light is modeled using a parametric bulge + disk + envelope [fixed to result of Search 1].
     - The lens galaxy mass is modeled using a total mass distribution [no prior initialization].
     - The source galaxy's light is a parametric bulge + disk + envelope [no prior initialization].
This search aims to accurately estimate the lens mass model and source model.
In search 3 of the SOURCE PARAMETRIC PIPELINE we fit a lens model where:
     - The lens galaxy light is modeled using a parametric bulge + disk + envelope [priors are not initialized from
     previous searches].
     - The lens galaxy mass is modeled using a total mass distribution [priors initialized from search 2].
     - The source galaxy's light is a parametric bulge + disk + envelope [priors initialized from search 2].
This search aims to accurately estimate the lens light model, mass model and source model.
-----------
Parameters for source_parametric.with_lens_light
-----------
analysis
setup_hyper
lens_bulge
lens_disk
lens_envelope
mass
shear
bulge_priormodel
source_bulge
source_disk
source_envelope
redshift_lens
redshift_source
mass_center
'''

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

bulge = af.Model(al.lp.EllSersic)
disk = af.Model(al.lp.EllExponential)
#bulge.centre = disk.centre -> Not true for SLACS3

#Will make preliminary mass model constraints using given values from Nightingale
e_comps = al.convert.elliptical_comps_from(axis_ratio=0.68, angle=111.7)
mass = af.Model(al.mp.EllIsothermal)
mass.einstein_radius=1.52
mass.elliptical_comps=e_comps

#########################################
#
#Defining constraints here from the file
#
########################################

source_parametric_results = slam.source_parametric.with_lens_light(
    settings_autofit=settings_autofit,
    analysis=analysis,
    setup_hyper=setup_hyper,
    lens_bulge=bulge,
    lens_disk=disk,
    mass=mass,
    shear=af.Model(al.mp.ExternalShear),
    source_bulge=af.Model(al.lp.EllSersic),
    mass_centre=(0.0, 0.0),
    redshift_lens=redshift_lens,
    redshift_source=redshift_source,
)

'''
SOURCE INVERSION PIPELINE
The SOURCE INVERSION PIPELINE (with lens light) uses four searches to initialize a robust model for the `Inversion`
that reconstructs the source galaxy's light. It begins by fitting a `VoronoiMagnification` pixelization with `Constant`
regularization, to set up the model and hyper images.
In search 1 of the SOURCE INVERSION PIPELINE we fit a lens model where:
    - The lens galaxy light is modeled using a parametric bulge + disk + envelope [parameters fixed to result of SOURCE
    PARAMETER PIPELINE].
     - The lens galaxy mass is modeled using a total mass distribution [parameters fixed to result of SOURCE PARAMETRIC
     PIPELINE].
     - The source galaxy's light is a `VoronoiMagnification` pixelization and `Constant` regularization scheme.
This search aims to quickly estimate values for the pixelization resolution and regularization coefficient.
In search 2 of the SOURCE INVERSION PIPELINE we fit a lens model where:
    - The lens galaxy light is modeled using a parametric bulge + disk + envelope [parameters fixed to result of SOURCE
    PARAMETER PIPELINE].
     - The lens galaxy mass is modeled using a total mass distribution [parameters initialized from the results of the
     SOURCE PARAMETRIC PIPELINE].
     - The source galaxy's light is a `VoronoiMagnification` pixelization and `Constant` regularization scheme
     [parameters are fixed to the result of search 1].
This search aims to improve the lens mass model using the search 1 `Inversion`.
In search 3 of the SOURCE INVERSION PIPELINE we fit a lens model where:
    - The lens galaxy light is modeled using a parametric bulge + disk + envelope [parameters fixed to result of SOURCE
    PARAMETER PIPELINE].
     - The lens galaxy mass is modeled using a total mass distribution [parameters fixed to result of search 2].
     - The source galaxy's light is the input pixelization and regularization.
This search aims to estimate values for the pixelization and regularization scheme.
In search 4 of the SOURCE INVERSION PIPELINE we fit a lens model where:
    - The lens galaxy light is modeled using a parametric bulge + disk + envelope [parameters fixed to result of SOURCE
    PARAMETER PIPELINE].
     - The lens galaxy mass is modeled using a total mass distribution [parameters initialized from the results of the
     search 2].
     - The source galaxy's light is the input pixelization and regularization scheme [parameters fixed to the result
     of search 3].
This search aims to improve the lens mass model using the input `Inversion`.
 - Uses a `VoronoiBrightnessImage` pixelization.
 - Uses an `AdaptiveBrightness` regularization.
 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PARAMETRIC PIPELINE through to the
 SOURCE INVERSION PIPELINE.
 Positions: We update the positions and positions threshold using the previous model-fitting result
 to remove unphysical solutions from the `Inversion` model-fitting.
'''

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

'''
LIGHT PARAMETRIC PIPELINE
The LIGHT PARAMETRIC PIPELINE uses one search to fit a complex lens light model to a high level of accuracy, using the
lens mass model and source light model fixed to the maximum log likelihood result of the SOURCE INVERSION PIPELINE.
- Uses a parametric `EllSersic` bulge and `EllExponential` disk with centres aligned for the lens galaxy's
 light [Do not use the results of the SOURCE PARAMETRIC PIPELINE to initialize priors].
- Uses an `EllIsothermal` model for the lens's total mass distribution [fixed from SOURCE PARAMETRIC PIPELINE].
- Uses an `Inversion` for the source's light [priors fixed from SOURCE INVERSION PIPELINE].
- Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PIPELINE through to the MASS
 PIPELINE [fixed values].
----------------
Parameters for light_parametric.with_lens_light
----------------
settings_autofit
analysis
setup_hyper
source_results
    The results of the SLaM SOURCE PARAMETRIC PIPELINE or SOURCE INVERSION PIPELINE which ran before this pipeline.
lens_bulge
lens_disk
lens_envelope
    set to None to omit an envelope.
end_with_hyper_extension
        If `True` a hyper extension is performed at the end of the pipeline. If this feature is used, you must be
        certain you have manually passed the new hyper images geneted in this search to the next pipelines.
'''

analysis = al.AnalysisImaging(
    dataset=imaging, hyper_dataset_result=source_inversion_results.last
)

bulge = af.Model(al.lp.EllSersic)
disk = af.Model(al.lp.EllExponential)
#bulge.centre = disk.centre -> Not applicable to SLACS3

light_results = slam.light_parametric.with_lens_light(
    settings_autofit=settings_autofit,
    analysis=analysis,
    setup_hyper=setup_hyper,
    source_results=source_inversion_results,
    lens_bulge=bulge,
    lens_disk=disk,
)

'''
MASS LIGHT DARK PIPELINE
The MASS LIGHT DARK PIPELINE (with lens light) uses one search to fits a complex lens mass model to a high level of
accuracy, using the source model of the SOURCE PIPELINE and the lens light model of the LIGHT PARAMETRIC PIPELINE to
initialize the model priors
 - Uses a parametric `EllSersic` bulge and `EllExponential` disk with centres aligned for the lens galaxy's
 light and its stellar mass [12 parameters: fixed from LIGHT PARAMETRIC PIPELINE].
 - The lens galaxy's dark matter mass distribution is a `SphNFW` whose centre is aligned with bulge of
 the light and stellar mass model above [5 parameters].
 - Uses an `Inversion` for the source's light [priors fixed from SOURCE INVERSION PIPELINE].
 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PARAMETRIC PIPELINE through to the MASS
 LIGHT DARK PIPELINE.
 - Hyper: We may be using hyper features and therefore pass the result of the SOURCE INVERSION PIPELINE to use as the
 hyper dataset if required.
 - Positions: We update the positions and positions threshold using the previous model-fitting result
  to remove unphysical solutions from the `Inversion` model-fitting.
'''

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
