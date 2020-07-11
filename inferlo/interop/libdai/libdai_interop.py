# Copyright (c) 2020, The InferLO authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 - see LICENSE file.
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Dict

import numpy as np

from inferlo.base.factors import DiscreteFactor
from inferlo.pairwise import InferenceResult

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

    So far it only uses belief propagation. libDAI supports more algorithms,
    support for which will be added later.
    """

    def __init__(self):
        import inferlo.interop.libdai as libdai_module
        path = Path(libdai_module.__file__).parent
        self.bin_path = os.path.join(path, 'bin')
        self.tmp_path = os.path.join(path, 'tmp')
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
        values = factor.values.transpose(rev_perm)
        flat_values = values.reshape(-1)
        for i in range(len(flat_values)):
            if abs(flat_values[i]) > 1e-9:
                value_lines.append("%d %.10f" % (i, flat_values[i]))
        return header_lines + [str(len(value_lines))] + value_lines

    def infer(self, model: GraphModel, algorithm="BP",
              options=None) -> InferenceResult:
        """Infrence with libDAI.

        Calculates partition function and marginal probabilities.
        """
        if options is None:
            options = dict()
        self._run(model, 'infer', algorithm, options)
        log_z = float(self.stdout)
        marg_probs = np.loadtxt(self.output_file)
        return InferenceResult(log_z, marg_probs)

    def max_likelihood(self, model: GraphModel, algorithm="BP",
                       options=None) -> np.ndarray:
        """Calculates most likely state with libDAI."""
        if options is None:
            options = dict()
        self._run(model, 'max_likelihood', algorithm, options)
        return np.loadtxt(self.output_file, dtype=np.int32)

    def _run(self, model: GraphModel, problem: str, algorithm: str,
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

        if process.returncode != 0:
            msg = process.stderr.read().decode("utf-8")
            raise ValueError("libDAI failed with error message: %s" % msg)
        self.stdout = process.stdout.read().decode("utf-8")
