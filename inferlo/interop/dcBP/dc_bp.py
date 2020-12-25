from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inferlo import GraphModel


def dc_bp(model: GraphModel, lib_path: str, machines=2, command="-e 0.0"):
    """ Interoperation with dcBP:
    https://www.alexander-schwing.de/projects.php

    For the algorithm see:
    www.alexander-schwing.de/papers/SchwingEtAl_CVPR2011a_IEEEeXpress.pdf

    :param model: input graphical model
    :param lib_path: path to a folder with compiled dcBP
    :param machines: number of local machines
    :param command: specify any other input commands
            for more commands run ./dcBP -h
    :return: list of beliefs
    """

    # check if allows to convert GraphModel into dcBP input format
    if ("save_uai" not in dir(model)):
        print("This type of graphical model does not support",
              "converting to UIA format yet")
    else:
        model.save_uai(lib_path, name="output")

        dcbp_command = "mpiexec -n " + str(machines)
        dcbp_command += " ./dcBP -f output.uai " + command
        dcbp_command += " -o result.txt"

        os.chdir(lib_path)
        os.system(dcbp_command)
        dcbp_result = open(lib_path+"/result.txt", "rb")
        beliefs = list(dcbp_result.read())

        return beliefs
