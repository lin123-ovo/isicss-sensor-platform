"""Fixed 16-member TabICL v2 crossline recipe."""
from . import crossline_compact36 as compact
from . import crossline_tabicl_single as original
from .tabicl_options import normalize_recipe

MODEL_NAME = 'tabicl_v2_single36_16'
# The feature contract and decoder are identical to the existing single model.
FEATURE_VERSION = original.FEATURE_VERSION
BASE_FEATURE_COLUMNS = compact.BASE_FEATURE_COLUMNS
FEATURE_COLUMNS = compact.FEATURE_COLUMNS
features_for = compact.features_for
history_for = compact.history_for
FROZEN_PARAMETERS = {**original.FROZEN_PARAMETERS, 'model': MODEL_NAME, 'n_estimators': 16}


def normalize_parameters(document):
    """Reject changes to the fixed, separately archived 16-member recipe."""
    return normalize_recipe(document, FROZEN_PARAMETERS)


def recipe_metadata(parameters=None):
    """Describe the actual single branch and its fixed ensemble size."""
    metadata = original.recipe_metadata()
    metadata['model'] = MODEL_NAME
    metadata['base_recipe']['parameters']['n_estimators'] = 16
    metadata['base_recipe']['tabicl']['n_estimators'] = 16
    if parameters is not None and parameters.get('recipe_version') == 2:
        values = normalize_parameters(parameters)
        metadata.update(parameter_version=2, context_fraction=values['context_fraction'],
                        temperature_calibration=values['temperature_calibration'],
                        context_selector='temperature_stable_spread_v1', calibration_policy='inner80_20_purge24h_nll_v1')
    return metadata
