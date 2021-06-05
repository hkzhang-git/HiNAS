from .dn_supernet import Dn_supernet
from .dn_compnet import Dn_compnet
from .sid_supernet import Sid_supernet
from .sid_compnet import Sid_compnet
from .sr_supernet import Sr_supernet
from .sr_compnet import Sr_compnet


ARCHITECTURES = {
    "Dn_supernet": Dn_supernet,
    "Dn_compnet": Dn_compnet,
    "Sid_supernet": Sid_supernet,
    "Sid_compnet": Sid_compnet,
    "Sr_supernet": Sr_supernet,
    "Sr_compnet": Sr_compnet
}


def build_model(cfg):
    meta_arch = ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
