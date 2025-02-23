"""
Custom Callbacks
"""
import lightning


class TorchGlobals(lightning.pytorch.callbacks.Callback):
    """
    Callback to setup torch globals

    Args:
        float32_matmul_precision (str):
            can be 'medium', 'high', 'default', or 'auto'.
            The 'default' value does not change any setting.
            The 'auto' value defaults to 'medium' if the training devices have
                ampere cores.
    """

    def __init__(self, float32_matmul_precision='default'):
        self.float32_matmul_precision = float32_matmul_precision

    def setup(self, trainer, pl_module, stage):
        import torch
        float32_matmul_precision = self.float32_matmul_precision
        if float32_matmul_precision == 'default':
            float32_matmul_precision = None
        elif float32_matmul_precision == 'auto':
            # Detect if we have Ampere tensor cores
            # Ampere (V8) and later leverage tensor cores, where medium
            # float32_matmul_precision becomes useful
            if torch.cuda.is_available():
                device_versions = [torch.cuda.get_device_capability(device_id)[0]
                                   for device_id in trainer.device_ids]
                if all(v >= 8 for v in device_versions):
                    float32_matmul_precision = 'medium'
                else:
                    float32_matmul_precision = None
            else:
                float32_matmul_precision = None
        if float32_matmul_precision is not None:
            torch.set_float32_matmul_precision(float32_matmul_precision)
