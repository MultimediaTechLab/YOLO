import lightning
import ubelt as ub


class YoloTrainer(lightning.Trainer):
    """
    Simple trainer subclass so we can ensure a print happens directly before
    the training loop.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hacked_torch_global_callback = TorchGlobals(float32_matmul_precision='auto')

    def _run(self, *args, **kwargs):
        # All I want is to print this  directly before training starts.
        # Is that so hard to do?
        self._on_before_run()
        super()._run(*args, **kwargs)

    def _run_stage(self, *args, **kwargs):
        # All I want is to print this  directly before training starts.
        # Is that so hard to do?
        self._on_before_run_stage()
        super()._run_stage(*args, **kwargs)

    @property
    def log_dpath(self):
        """
        Get path to the the log directory if it exists.
        """
        if self.logger is None:
            # Fallback to default root dir
            return ub.Path(self.default_root_dir)
            # raise Exception('cannot get a log_dpath when no logger exists')
        if self.logger.log_dir is None:
            return ub.Path(self.default_root_dir)
            # raise Exception('cannot get a log_dpath when logger.log_dir is None')
        return ub.Path(self.logger.log_dir)

    def _on_before_run(self):
        """
        Our custom "callback"
        """
        self._hacked_torch_global_callback.before_setup_environment(self)

    def _on_before_run_stage(self):
        """
        Our custom "callback"
        """
        print(f'self.global_rank={self.global_rank}')
        if self.global_rank == 0:
            self._on_before_run_rank0()

    def _on_before_run_rank0(self):
        import rich
        dpath = self.log_dpath
        rich.print(f"Trainer log dpath:\n\n[link={dpath}]{dpath}[/link]\n")


class TorchGlobals(lightning.pytorch.callbacks.Callback):
    """
    Callback to setup torch globals.

    Note: this needs to be called before the accelerators are setup, and
    existing callbacks don't have mechanisms for that, so we hack it in here.

    Args:
        float32_matmul_precision (str):
            can be 'medium', 'high', 'default', or 'auto'.
            The 'default' value does not change any setting.
            The 'auto' value defaults to 'medium' if the training devices have
                ampere cores.
    """

    def __init__(self, float32_matmul_precision='default'):
        self.float32_matmul_precision = float32_matmul_precision

    def before_setup_environment(self, trainer):
        import torch
        print('Setup Torch Globals')
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
            print(f'Update: float32_matmul_precision={float32_matmul_precision}')
            torch.set_float32_matmul_precision(float32_matmul_precision)
