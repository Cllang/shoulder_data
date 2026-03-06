from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.nets.UxLSTMEnc_3d import get_uxlstm_enc_3d_from_plans
from nnunetv2.nets.UxLSTMEnc_2d import get_uxlstm_enc_2d_from_plans

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice_focal_loss import DiceFocalBCELoss
import numpy as np

class nnUNetTrainerUxLSTMEnc(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        if len(configuration_manager.patch_size) == 2:
            model = get_uxlstm_enc_2d_from_plans(plans_manager, dataset_json, configuration_manager,
                                          num_input_channels, deep_supervision=enable_deep_supervision)
        elif len(configuration_manager.patch_size) == 3:
            model = get_uxlstm_enc_3d_from_plans(plans_manager, dataset_json, configuration_manager,
                                          num_input_channels, deep_supervision=enable_deep_supervision)
        else:
            raise NotImplementedError("Only 2D and 3D models are supported")

        
        print("UxLSTMEnc: {}".format(model))

        return model


    def _build_loss(self):
        self.print_to_log_file("Using Dice + BCE + Focal Loss for training.")

        # Define the loss function
        loss = DiceFocalBCELoss()  # Adjust parameters if needed

        # Apply Deep Supervision if enabled
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

            # Avoid errors with Distributed Data Parallel (DDP)
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6  # Prevent unused parameter issues
            else:
                weights[-1] = 0  # Last output has no weight

            # Normalize weights
            weights = weights / weights.sum()

            # Wrap with Deep Supervision
            loss = DeepSupervisionWrapper(loss, weights)

        return loss
