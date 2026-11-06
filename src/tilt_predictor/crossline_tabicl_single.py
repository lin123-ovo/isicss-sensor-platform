"""Fixed independent classifier recipe for crossline ablation."""
from . import crossline_compact36 as compact
from .tabicl_options import normalize_recipe

MODEL_NAME = 'tabicl_v2_single36'
FEATURE_VERSION = 'tilt_temperature_tabicl_single36_v1'
BASE_FEATURE_COLUMNS = compact.BASE_FEATURE_COLUMNS
FEATURE_COLUMNS = compact.FEATURE_COLUMNS
features_for = compact.features_for
history_for = compact.history_for
FROZEN_PARAMETERS = {**{k: v for k, v in compact.FROZEN_PARAMETERS.items() if k not in ('tabicl_weight', 'tabm_weight')}, 'model': MODEL_NAME}


def normalize_parameters(document):
    return normalize_recipe(document, FROZEN_PARAMETERS)


def recipe_metadata(parameters=None):
    result = {'model': MODEL_NAME, 'feature_version': FEATURE_VERSION, 'branch': 'tabicl',
            'base_recipe': compact.recipe_metadata(), 'target_observations_at_prediction': False}
    if parameters is not None and parameters.get('recipe_version') == 2:
        values = normalize_parameters(parameters)
        result.update(parameter_version=2, context_fraction=values['context_fraction'],
                      temperature_calibration=values['temperature_calibration'],
                      context_selector='temperature_stable_spread_v1', calibration_policy='inner80_20_purge24h_nll_v1')
    return result
