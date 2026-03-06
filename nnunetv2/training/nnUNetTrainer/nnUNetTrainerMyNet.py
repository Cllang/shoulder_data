# 作者：Lang
# 2024年09月12日17时43分01秒
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.nets.MyNet import get_myNet_from_plans


class nnUNetTrainerMyNet(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        model = get_myNet_from_plans(plans_manager,
                                     dataset_json,
                                     configuration_manager,
                                     num_input_channels,
                                     deep_supervision=enable_deep_supervision)

        print("myNet: {}".format(model))
        return model

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        if self.is_ddp:
            self.network.module.deep_supervision = enabled
        else:
            self.network.deep_supervision = enabled
    def _get_deep_supervision_scales(self):
        if self.enable_deep_supervision:
            deep_supervision_scales = [[1.0,1.0,1.0], [0.5,0.5,0.5], [0.25,0.25,0.25],[0.125,0.125,0.125]]
        else:
            deep_supervision_scales = None  # for train and val_transforms
        return deep_supervision_scales

