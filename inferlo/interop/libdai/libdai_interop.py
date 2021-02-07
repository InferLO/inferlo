# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
from __future__ import annotations

import copy
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Dict

import numpy as np

from inferlo.base import InferenceResult
from inferlo.base.factors import DiscreteFactor

if TYPE_CHECKING:
    from inferlo import GraphModel


class LibDaiInterop():
    """ Interoperation with libDAI.

    See https://staff.fnwi.uva.nl/j.m.mooij/libDAI/

    Interoperation uses file interface. We created a simple C++ file which
    reads model from file, solves problem for it using libDAI and writes output
    to another file. This Python file is reponsible for converting model to
    libDAI format, writing input file, invoking libDAI and reading the output
    file.

    To use it, you will have to install libDAI on your own:

    1. Get libDAI:
    ``git clone https://bitbucket.org/jorism/libdai.git``.

    2. Build libDAI following instructions in README.
    3. Specify path to libdai dir in ``bin/build.sh``.
    4. Run ``build.sh``.

    This works on Ubuntu, wasn't checked for other platforms.

    We use libDAI only for verification and benchmarking of inferlo's own
    functionality. We don't use in any of inferlo's algorithms.

    Algorithms
        libDAI supports several inference algorithms, identified by codename.
        For full list of algorithms, see LibDaiInterop.ALL_ALGORITHMS.

        List of supported algorithms with their full names:

        * BP - (Loopy) Belief Propagation;
        * CBP - Conditioned Belief Propagation;
        * DECMAP - Decimation algorithm;
        * EXACT - Exact inference by brute force enumeration;
        * FBP - Fractional Belief Propagation;
        * GIBBS - Gibbs sampler;
        * GLC - Generalized Loop Corrected Belief Propagation;
        * HAK - Double-loop Generalized Belief Propagation;
        * JTREE - Exact inference by junction-tree methods;
        * LC - Loop Corrected Belief Propagation;
        * MF - Mean Field;
        * MR - Approximate inference algorithm by Montanari and Rizzo;
        * TREEEP - Tree Expectation Propagation;
        * TRWBP - Tree-Reweighted Belief Propagation;

        All algorithms support calculating marginal probabilties, not all of
        them support calculating logZ or most likely state. If algorithm
        doesn't support calculating logZ, you'll get 0. If algorithms doesn't
        support calculating most likely state, exception will be thrown.

        For documentation on particular algorithms, refer to libDAI
        documentation: https://staff.fnwi.uva.nl/j.m.mooij/libDAI/doc/.

    Options
        Almost every algorithm requires some parameters. Parameters are passed
        to algorithm as dict, where keys are string parameter names. Value can
        be number, enum value (represented by its name), or nested set of
        parameters - then it's encoded in format "[key1=val1,key2=val2]".

        For documentation on options for every particular algorithm, refer to
        libDAI documentation and look for Properties class nested in class
        implementing that inference algorithm.

        If not specified, default options will be used. You can access default
        options for all algorithms as LibDaiInterop.DEFAULT_OPTIONS. These
        options aren't "recommended", they just specify minimal set of options
        which makes all algorithms work.
    """

    #: All algorithms.
    ALL_ALGORITHMS = ['BP', 'CBP', 'DECMAP', 'EXACT', 'FBP', 'GIBBS', 'GLC',
                      'HAK', 'JTREE', 'LC', 'MF', 'MR', 'TREEEP', 'TRWBP']

    #: Algorithms which can be used for Maximum Likelihood problem.
    ML_ALGORITHMS = ['BP', 'DECMAP', 'EXACT', 'FBP', 'GIBBS', 'JTREE',
                     'TREEEP', 'TRWBP']

    #: Default options for all algorithms.
    DEFAULT_OPTIONS = {
        'EXACT': dict(),
        'MF': {'tol': 1e-9, 'maxiter': 100},
        'BP': {'updates': 'SEQRND', 'logdomain': 0, 'tol': 1e-9},
        'MR': {'updates': 'FULL', 'inits': 'RESPPROP', 'tol': 1e-9},
        'HAK': {'clusters': 'BETHE', 'doubleloop': 0, 'tol': 1e-9,
                'maxiter': 10000},
        'LC': {'tol': 1e-9, 'updates': 'SEQRND', 'cavity': 'FULL',
               'maxiter': 10000,
               'cavainame': 'BP',
               'cavaiopts': '[tol=1e-9,logdomain=0,updates=SEQRND]'},
        'JTREE': {'updates': 'HUGIN'},
        'DECMAP': {'ianame': 'BP',
                   'iaopts': '[tol=1e-9,logdomain=0,updates=SEQRND]'},
        'FBP': {'tol': 1e-9, 'logdomain': 0, 'updates': 'SEQFIX'},
        'GLC': {'maxiter': 10000, 'tol': 1e-9, 'cavity': 'FULL',
                'updates': 'SEQFIX', 'rgntype': 'SINGLE',
                'cavainame': 'BP',
                'cavaiopts': '[tol=1e-9,logdomain=0,updates=SEQRND]',
                'inainame': 'BP',
                'inaiopts': '[tol=1e-9,logdomain=0,updates=SEQRND]'},
        'GIBBS': {'maxiter': 10000},
        'TREEEP': {'tol': 1e-9, 'type': 'ORG'},
        'TRWBP': {'tol': 1e-9, 'updates': 'SEQFIX', 'logdomain': 0},
        'CBP': {'tol': 1e-9, 'updates': 'SEQFIX', 'maxiter': 10000,
                'rec_tol': 1e-9, 'min_max_adj': 123,
                'choose': 'CHOOSE_RANDOM', 'recursion': 'REC_FIXED',
                'clamp': 'CLAMP_VAR', 'bbp_props': '[]',
                'bbp_cfn': 'CFN_GIBBS_B'}}

    def __init__(self):
        import inferlo.interop.libdai as libdai_module
        path = Path(libdai_module.__file__).parent
        self.bin_path = os.path.join(path, 'bin')
        self.tmp_path = os.path.join(tempfile.gettempdir(), 'inferlo_libdai')
        if not os.path.exists(self.tmp_path):
            os.makedirs(self.tmp_path)
        self.exe_path = os.path.join(self.bin_path, 'run_libdai.exe')
        self.input_file = os.path.join(self.tmp_path, 'input.fg')
        self.output_file = os.path.join(self.tmp_path, 'output.txt')
        self.ready = False

        # Check that executable file can be executed.
        if os.path.exists(self.exe_path):
            try:
                process = subprocess.Popen(args=[self.exe_path, 'problem=x'],
                                           stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL)
                process.wait()
                self.ready = (process.returncode == 1)
            except OSError as err:
                print(err)

    def is_libdai_ready(self):
        """Checks whether LibDAI is ready to work.

        This means run_libdai executable is built and can be executed by OS.
        """
        return self.ready

    @staticmethod
    def write_fg_file(model: GraphModel, file_path: str):
        """ Writes model to FG file.

        See format definition at
        https://staff.fnwi.uva.nl/j.m.mooij/libDAI/doc/fileformats.html
        """
        factors = list(model.get_factors())
        assert all([f.is_discrete() for f in factors])
        factors = [DiscreteFactor.from_factor(f) for f in factors]

        vars_in_factors = set([i for f in factors for i in f.var_idx])
        all_vars = set(range(model.num_variables))
        unused_vars = all_vars - vars_in_factors
        if len(unused_vars) > 0:
            print(
                "Not all variables are referenced in factors, will add dummy "
                "unit factors for them. Unused variables: %s" % unused_vars)
            for var_id in unused_vars:
                size = model.get_variable(var_id).domain.size()
                factors.append(DiscreteFactor(model, [var_id], np.ones(size)))

        with open(file_path, "w") as file:
            file.write("%d\n\n" % len(factors))
            for f in factors:
                file.write('\n'.join(LibDaiInterop._to_fg_factor(f)))
                file.write("\n\n")

    @staticmethod
    def _to_fg_factor(factor: DiscreteFactor) -> List[str]:
        """Encodes discrete factor in libDAI FG format."""
        domain_sizes = [factor.model.get_variable(i).domain.size() for i in
                        factor.var_idx]
        header_lines = [str(len(factor.var_idx)),
                        ' '.join(map(str, factor.var_idx)),
                        ' '.join(map(str, domain_sizes))]

        value_lines = []
        rev_perm = list(range(len(factor.var_idx)))[::-1]
        flat_values = factor.values.transpose(rev_perm).reshape(-1)
        for i in range(len(flat_values)):
            if abs(flat_values[i]) > 1e-9:
                value_lines.append("%d %.10f" % (i, flat_values[i]))
        return header_lines + [str(len(value_lines))] + value_lines

    def infer(self, model: GraphModel, algorithm,
              options: Dict[str, str] = None) -> InferenceResult:
        """Inference with libDAI.

        Calculates logarithm of partition function and marginal probabilities.

        :param model: Model, for which to perform inference. All factors must
           be discrete.
        :param model: Inference algorithm. Must be one of
          ``LibDaiInterop.ALL_ALGORITHMS``.
        :param options: libDAI options.
        :return: InferenceResult containing log partition function and marginal
          probabilities.
        """
        assert algorithm in LibDaiInterop.ALL_ALGORITHMS
        if options is None:
            options = LibDaiInterop.DEFAULT_OPTIONS[algorithm]
        options = copy.deepcopy(options)
        self._run(model, 'infer', algorithm, options)
        log_z = float(self.stdout)
        marg_probs = np.loadtxt(self.output_file)
        if len(marg_probs.shape) == 1:
            marg_probs = np.array([marg_probs])
        return InferenceResult(log_z, marg_probs)

    def max_likelihood(self, model: GraphModel, algorithm,
                       options: Dict[str, str] = None) -> np.ndarray:
        """Calculates most likely state with libDAI.

        :param model: Model, for which to perform inference. All factors must
           be discrete.
        :param model: Inference algorithm. Must be one of
          ``LibDaiInterop.ML_ALGORITHMS``.
        :param options: libDAI options.
        :return: Marginal probabilities. Array of shape (number of variables,
          number of variables). If variables are from different domains, second
          dimension will be equal to maximal domain size, and for variables
          having less possible values, probabilities for impossible values will
          be padded with zeroes.
        """
        assert algorithm in LibDaiInterop.ML_ALGORITHMS
        if options is None:
            options = LibDaiInterop.DEFAULT_OPTIONS[algorithm]
        options = copy.deepcopy(options)
        options['inference'] = 'MAXPROD'
        self._run(model, 'max_likelihood', algorithm, options)
        return np.loadtxt(self.output_file, dtype=np.int32)

    def _run(self, model: GraphModel, problem: str, algorithm,
             options: Dict[str, str]):
        """Invokes run_libdai for given model and problem."""
        assert self.ready, "libDAI is not ready."
        LibDaiInterop.write_fg_file(model, self.input_file)

        encoded_options = ','.join(
            ["%s=%s" % (k, v) for k, v in options.items()])
        encoded_options = '[' + encoded_options + ']'

        start_time = time.time()
        process = subprocess.Popen(
            args=[self.exe_path, self.input_file, self.output_file,
                  problem, algorithm, encoded_options],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        process.wait()
        # Store true running time, which doesn't include conversions and IO.
        self.true_running_time = time.time() - start_time
        self.stdout = process.stdout.read().decode("utf-8")
        self.stderr = process.stderr.read().decode("utf-8")

        if process.returncode != 0:
            raise ValueError(
                "libDAI failed with error message: %s" %
                self.stderr)
