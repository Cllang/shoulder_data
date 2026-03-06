from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
import torch
from torch import nn
from nnunetv2.nets.SCFR_5 import get_SCFR_5_from_plans


class nnUNetTrainerSCFR_5(nnUNetTrainer):

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        print("Using nnUNetTrainerSCFR_5")
        super().__init__(
            plans, configuration, fold, dataset_json, unpack_dataset, device
        )
        self.num_epochs = 500

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        model = get_SCFR_5_from_plans(plans_manager,
                                     dataset_json,
                                     configuration_manager,
                                     num_input_channels,
                                     deep_supervision=enable_deep_supervision)

        print("SCFR_5: {}".format(model))
        return model