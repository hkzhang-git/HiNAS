from .denoise import Dn_datasets
# from .see_in_dark import Sid_dataset
from .super_resolution import Sr_datasets

tasks_dict = {
    'dn': Dn_datasets,
    # 'sid': Sid_dataset,
    'sr': Sr_datasets
}


