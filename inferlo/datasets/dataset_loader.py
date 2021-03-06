# Copyright (c) The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
import os
from dataclasses import dataclass
import tempfile
from typing import Dict

import numpy as np
import wget

from inferlo import GenericGraphModel
from inferlo.datasets.uai_reader import UaiReader
from inferlo.datasets.uai_writer import UaiWriter

REPO_URL = 'https://raw.githubusercontent.com/akxlr/tbp/master/tests/uai/'
UAI_PROB_URL = REPO_URL + 'MAR_prob'
UAI_SOL_URL = REPO_URL + 'MAR_sol'

# Copied from http://sli.ics.uci.edu/~ihler/uai-data/
# Values here are log10.
# TODO: load data from remote data repository.
UAI_PF = {
    'DBN_11.uai': 58.5307,
    'CSP_11.uai': 13.563,
    'CSP_12.uai': 16.4536,
    'CSP_13.uai': 15.3037,
    'Promedus_11.uai': -8.39145,
    'Promedus_12.uai': -3.16462,
    'Promedus_13.uai': -4.50703,
    'Promedus_14.uai': -7.80675,
    'Promedus_15.uai': -3.63667,
    'Promedus_16.uai': -6.97522,
    'Promedus_17.uai': -9.4592,
    'Promedus_18.uai': -4.67228,
    'Promedus_19.uai': -4.34464,
    'Promedus_20.uai': -7.06065,
    'Promedus_21.uai': -5.58012,
    'Promedus_22.uai': -2.49474,
    'Promedus_23.uai': -11.3151,
    'Promedus_24.uai': -5.86181,
    'Promedus_25.uai': -9.41791,
    'Promedus_26.uai': -7.30489,
    'Promedus_27.uai': -8.13576,
    'Promedus_28.uai': -8.13152,
    'Promedus_29.uai': -10.4527,
    'Promedus_30.uai': -22.1005,
    'Promedus_31.uai': -1.79979,
    'Promedus_32.uai': -2.20193,
    'Promedus_33.uai': -2.80866,
    'Promedus_34.uai': -3.07996,
    'Promedus_35.uai': -1.74295,
    'Promedus_36.uai': -1.7427,
    'Promedus_37.uai': -4.1711,
    'Promedus_38.uai': -4.98235,
    'linkage_11.uai': -31.1776,
    'linkage_12.uai': -69.7017,
    'linkage_13.uai': -76.0389,
    'linkage_14.uai': -30.7614,
    'linkage_15.uai': -76.6058,
    'linkage_16.uai': -38.5556,
    'linkage_17.uai': -64.8245,
    'linkage_18.uai': -78.5222,
    'linkage_19.uai': -59.0249,
    'linkage_20.uai': -64.2292,
    'linkage_21.uai': -29.6304,
    'linkage_22.uai': -54.2777,
    'linkage_23.uai': -116.58,
    'linkage_24.uai': -83.7331,
    'linkage_25.uai': -87.8848,
    'linkage_26.uai': -115.772,
    'linkage_27.uai': -63.4735,
}


@dataclass
class Dataset:
    """Dataset containing GM and true answers.

    Fields:
        * ``model`` - Graphical model.
        * ``true_marginals`` - known true exact marginal probabilities for all variables.
        * ``true_log_pf`` - known true exact natural logarithm of the partition function.
        * ``name`` - name of the dataset.
    """
    model: GenericGraphModel
    true_marginals: np.array
    true_log_pf: float
    name: str


class DatasetLoader:
    """Loads graphical models from named datasets."""

    def __init__(self, data_dir=None):
        """
        :param data_dir: Where to store cached datasets. Specify if you want
          datasets being cached locally. If not set, default system temporary
          directory will be used.
        """
        if data_dir is None:
            data_dir = os.path.join(tempfile.gettempdir(), 'inferlo_data')
        self.data_dir = os.path.expanduser(data_dir)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.custom_dir = os.path.join(self.data_dir, 'custom')
        if not os.path.exists(self.custom_dir):
            os.makedirs(self.custom_dir)
        self.uai_reader = UaiReader()
        self.uai_writer = UaiWriter()

    def load_file_(self, url_prefix, file_name):
        """Loads file from the web to local file.

        If local file already exists, does nothing.
        Returns local path to loaded file.
        """
        path = os.path.join(self.data_dir, file_name)
        if not os.path.exists(path):
            url = url_prefix + '/' + file_name
            wget.download(url, out=path)
            assert os.path.exists(path)
        return path

    def load_uai_dataset(self, dataset_name) -> Dataset:
        """Loads named dataset from UAI competition.

        :param dataset_name: Name of dataset, e.g. "Promedus_11.uai". For full
            list of UAI datasets, see http://sli.ics.uci.edu/~ihler/uai-data/.
            Not all of them are currently supported.
        :return: `Dataset` object, containing graphical model
            (as `GenericGraphModel` object), true logarithm of the partition
            function and true marginals.
        """
        if dataset_name not in UAI_PF:
            raise ValueError("Unknown UAI dataset: " + dataset_name)
        data_file = self.load_file_(UAI_PROB_URL, dataset_name)
        model = self.uai_reader.read_model(data_file)

        # True marginals.
        mar_file = self.load_file_(UAI_SOL_URL, dataset_name + '.MAR')
        marginals = self.uai_reader.read_marginals(mar_file)
        return Dataset(model=model,
                       true_marginals=marginals,
                       true_log_pf=UAI_PF[dataset_name] * np.log(10),
                       name=dataset_name)

    def save_custom_dataset(self, dataset):
        """Saves dataset locally."""
        # Store model.
        self.uai_writer.write_model(dataset.model,
                                    os.path.join(self.custom_dir, dataset.name + '.uai'))

        # Store true answers.
        ans_path = os.path.join(self.custom_dir, dataset.name + '.ans.npy')
        ans = {'true_marginals': dataset.true_marginals, 'true_log_pf': dataset.true_log_pf}
        np.save(ans_path, ans)

    def load_custom_dataset(self, name: str) -> Dataset:
        """Loads previously saved datataset."""
        model_path = os.path.join(self.custom_dir, name + '.uai')
        assert os.path.exists(model_path)
        model = self.uai_reader.read_model(model_path)

        ans_path = os.path.join(self.custom_dir, name + '.ans.npy')
        assert os.path.exists(model_path)
        ans = np.load(ans_path, allow_pickle=True).item()  # type: Dict
        return Dataset(model=model,
                       true_log_pf=ans['true_log_pf'],
                       true_marginals=ans['true_marginals'],
                       name=name)

    def custom_dataset_exists(self, name: str):
        """Checks if custom dataset exists."""
        return os.path.exists(os.path.join(self.custom_dir, name + '.uai'))
