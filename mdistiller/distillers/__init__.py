from ._base import Vanilla
from .KD import KD
from .AT import AT
from .OFD import OFD
from .RKD import RKD
from .FitNet import FitNet
from .KDSVD import KDSVD
from .CRD import CRD
from .NST import NST
from .PKT import PKT
from .SP import SP
from .VID import VID
from .ReviewKD import ReviewKD
from .DKD import DKD
from .TECH_KD import Tech_KD
from .TECH_DKD import Tech_DKD
from .TECH_FITNET import Tech_FitNet
from .DFGKD import DFGKD
from.TECH_AT import Tech_AT
distiller_dict = {
    "NONE": Vanilla,
    "KD": KD,
    "AT": AT,
    "OFD": OFD,
    "RKD": RKD,
    "FITNET": FitNet,
    "KDSVD": KDSVD,
    "CRD": CRD,
    "NST": NST,
    "PKT": PKT,
    "SP": SP,
    "VID": VID,
    "REVIEWKD": ReviewKD,
    "DKD": DKD,
    "TECH_KD": Tech_KD,
    "TECH_DKD": Tech_DKD,
    "TECH_AT": Tech_AT,
    "TECH_FITNET": Tech_FitNet,
    "DFGKD": DFGKD,
}
