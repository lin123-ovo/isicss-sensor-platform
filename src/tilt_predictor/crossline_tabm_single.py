"""Fixed independent classifier recipe for crossline ablation."""
from . import crossline_compact36 as compact

MODEL_NAME = 'tabm_single36'
FEATURE_VERSION = 'tilt_temperature_tabm_single36_v1'
BASE_FEATURE_COLUMNS = compact.BASE_FEATURE_COLUMNS
FEATURE_COLUMNS = compact.FEATURE_COLUMNS
features_for = compact.features_for
history_for = compact.history_for
FROZEN_PARAMETERS = {**{k: v for k, v in compact.FROZEN_PARAMETERS.items() if k not in ('tabicl_weight', 'tabm_weight')}, 'model': MODEL_NAME}


def normalize_parameters(document):
    if not isinstance(document, dict) or set(document) != set(FROZEN_PARAMETERS) or any(isinstance(document[k], bool) or document[k] != v for k, v in FROZEN_PARAMETERS.items()):
        raise ValueError('单模型需要完整固定配置。')
    return dict(FROZEN_PARAMETERS)


def recipe_metadata():
    return {'model': MODEL_NAME, 'feature_version': FEATURE_VERSION, 'branch': 'tabm',
            'base_recipe': compact.recipe_metadata(), 'target_observations_at_prediction': False}
