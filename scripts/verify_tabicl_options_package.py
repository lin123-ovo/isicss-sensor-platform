"""Frozen-runtime synthetic TabICL fit/archive/predict checks; no database access."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd

from sensor_app.model_assets import TABICL_CHECKPOINT_SHA256
from tilt_predictor import crossline_compact36 as compact
from tilt_predictor.crossline_single import SingleBranchRegressor


def verify() -> dict:
    """Load the bundled checkpoint and round-trip both member counts and flags."""
    frozen = bool(getattr(sys, 'frozen', False))
    if frozen:
        module_path = Path(sys.modules[SingleBranchRegressor.__module__].__file__).resolve()
        if not module_path.is_relative_to(Path(sys.executable).resolve().parents[1]):
            raise ValueError('冻结检查意外使用了App之外的模型代码。')
    rng = np.random.default_rng(46)
    x = rng.normal(size=(1000, 34))
    x[:, 2] = 25+np.sin(np.arange(len(x))/30)
    y = compact.legacy.VALUES[np.arange(len(x)) % 9]
    times = pd.date_range('2026-01-01', periods=len(x), freq='10min')
    checks = []
    with TemporaryDirectory(prefix='isicss-tabicl-options-') as directory:
        for members in (4, 16):
            for enabled in (False, True):
                target = y
                if enabled:
                    labels = np.arange(len(x)) % 8
                    labels[-10:] = 8
                    target = compact.legacy.VALUES[labels]
                model = SingleBranchRegressor('tabicl', members, context_fraction=.5,
                    temperature_calibration=enabled, parameter_version=2).fit(x, target, timestamps=times)
                model.history_frame_ = pd.DataFrame({'timestamp': times, 'tilt1_inline_model': x[:, 0],
                    'tilt1_crossline_model': x[:, 1], 'temperature_celsius': x[:, 2]})
                model.training_info_['history_rows'] = len(times)
                model.validate_state()
                expected = model.predict(x[-4:])
                path = Path(directory)/'model.joblib'
                joblib.dump(model, path)
                restored = joblib.load(path)
                np.testing.assert_array_equal(expected, restored.predict(x[-4:]))
                if restored.training_info_['context_n'] != 500 or not np.isin(expected, compact.legacy.VALUES).all():
                    raise ValueError('冻结程序上下文或预测合同不一致。')
                calibration = restored.training_info_['calibration']
                if enabled and (calibration.get('calibration_used_n') != 190
                                or calibration.get('calibration_uncovered_n') != 10
                                or calibration.get('uncovered_class_counts') != [{'class': 8, 'count': 10}]):
                    raise ValueError('冻结程序未正确处理校准期的新档位。')
                checks.append({'members': members, 'calibration': enabled,
                               'temperature': restored.calibrated_temperature_, 'context_n': 500,
                               'calibration_used_n': calibration.get('calibration_used_n', 0),
                               'calibration_uncovered_n': calibration.get('calibration_uncovered_n', 0),
                               'roundtrip': True})
                del model, restored
                compact.release_device()
    return {'passed': True, 'frozen': frozen, 'bundled_model_code': frozen,
            'checkpoint_sha256': TABICL_CHECKPOINT_SHA256, 'checkpoint_loaded': True,
            'database_accessed': False, 'checks': checks}


if __name__ in ('__main__', 'bundle_validation'):
    destination = Path(os.environ.get('ISICSS_TABICL_OPTIONS_REPORT', 'reports/verification/tabicl_options_package.json'))
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = verify()
    except Exception as error:
        destination.write_text(json.dumps({'passed': False, 'error': str(error)}, ensure_ascii=False, indent=2))
        raise
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2))
