from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
import torch
from torch import nn
from nnunetv2.nets.BotaMSFR import get_BotaMSFR_from_plans


class nnUNetTrainerBotaMSFR(nnUNetTrainer):

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        print("Using nnUNetTrainerBotaMSFR")
        super().__init__(
            plans, configuration, fold, dataset_json, unpack_dataset, device
        )
        self.num_epochs = 400

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        model = get_BotaMSFR_from_plans(plans_manager,
                                     dataset_json,
                                     configuration_manager,
                                     num_input_channels,
                                     deep_supervision=enable_deep_supervision)

        print("BotaMSFR: {}".format(model))
        return model
