from basemodel.Baseline import Baseline

from DA_MD.L3B import L3B
from DA_MD.L2B import L2B

from DG.DANN_DG import DANN_DG
from DG.CORAL_DG import CORAL_DG
from DG.RSC import RSC
from DG.VREx import VREx

from DA.DANN_DA import DANN_DA
from DA.CORAL_DA import CORAL_DA
from DA.DAAN import DAAN
from DA.DSAN import DSAN


def get_model(name):
    if name not in globals():
        raise NotImplementedError(
            "Algorithm not found: {}".format(name))
    return globals()[name]

def print_args(args, print_list):
    s=""
    l = len(print_list)
    for arg, content in args.__dict__.items():
        if l == 0 or arg in print_list:
            s += "{}: {}\n".format(arg, content)
    return s