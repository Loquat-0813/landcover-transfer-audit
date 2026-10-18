"""Two fixed information arms; no trainable backbone or feature replacement."""
import torch
from p1backbone.hf_clip import FrozenCLIPPatchEncoder
from p1backbone.dense_readout import encode_readouts
from p1model.dense_residual import ARMS


@torch.inference_mode()
def pack_information(backbone, images, arm):
    if arm not in ARMS:
        raise ValueError('unknown information arm')
    if arm == 'native_info':
        native = FrozenCLIPPatchEncoder.encode_image(backbone, images)
        auxiliary = native
    else:
        readouts = encode_readouts(backbone, images)
        native, auxiliary = readouts['native'], readouts['dense_qq']
    return torch.cat((native, auxiliary), dim=1)


class DualReadoutEncoder(FrozenCLIPPatchEncoder):
    def __init__(self, model_id, arm):
        super().__init__(model_id)
        if arm not in ARMS:
            raise ValueError('unknown information arm')
        self.arm = arm

    @torch.inference_mode()
    def encode_image(self, images):
        return pack_information(self, images, self.arm)
