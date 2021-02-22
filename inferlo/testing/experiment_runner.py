# Copyright (c) The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE.
import json
import os
import tempfile
import time
from typing import Dict, Callable


class ExperimentRunner:
    """Helper class to run parametrized experiments.

    Experiment is a function with named arguments, which returns serializable
    object (e.g. number, or dict).

    It stores results locally in text files in system temporary directory. They
    are not lost even after reboot. This helps to run expensive experiments once
    and then process them later.
    """

    def __init__(self, data_dir: str = None):
        """
        :param data_dir: Where to store cached results.
        """
        if data_dir is None:
            data_dir = os.path.join(tempfile.gettempdir(),
                                    'inferlo_experiments')
        self.data_dir = os.path.expanduser(data_dir)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def _load_result(self, file_path: str):
        with open(os.path.join(self.data_dir, file_path), 'r') as f:
            return json.loads(f.read())

    def run_experiment(self, func: Callable, params: Dict):
        """Run experiment with given parameters.

        If experiment have already been run, returns results from cache.
        """
        params_list = []
        for key, val in params.items():
            assert isinstance(key, str)
            param = key + '=' + str(val)
            assert len(param.split('=')) == 2
            assert '&' not in param
            params_list.append(param)
        params_list.sort()
        directory = os.path.join(self.data_dir, func.__name__)
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_name = '&'.join(params_list) + '.dat'
        file_path = os.path.join(directory, file_name)
        if os.path.exists(file_path):
            # Experiment has already been run.
            return self._load_result(file_path)
        else:
            time_start = time.time()
            results = func(**params)
            elapsed_time = time.time() - time_start
            if not isinstance(results, dict):
                results = {'result': results}
            assert isinstance(results, dict)
            results['elapsed_time'] = elapsed_time
            with open(file_path, 'w') as f:
                f.write(json.dumps(results))
            return results

    def get_results(self, exp_name: str, filter_params: Dict = None):
        """Returns all results of given experiment with different parameters."""

        def _get_value(value: str):
            try:
                return int(value)
            except BaseException:
                try:
                    return float(value)
                except BaseException:
                    return value

        def _params_match(params):
            if filter_params:
                for key, value in filter_params.items():
                    if key not in params:
                        return False
                    if not params[key] == value:
                        return False
            return True

        directory = os.path.join(self.data_dir, exp_name)
        if not os.path.exists(directory):
            return []
        results = []
        for file_name in os.listdir(directory):
            assert file_name[-4:] == '.dat'
            params = {}
            if len(file_name) > 4:
                tokens = file_name[:-4].split('&')
                tokens = [t.split('=') for t in tokens]
                params = {t[0]: _get_value(t[1]) for t in tokens}
            if _params_match(params):
                result = self._load_result(os.path.join(directory, file_name))
                results.append({'params': params, 'result': result})
        return results
