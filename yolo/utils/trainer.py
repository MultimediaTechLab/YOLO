import lightning
import ubelt as ub


class YoloTrainer(lightning.Trainer):
    """
    Simple trainer subclass so we can ensure a print happens directly before
    the training loop.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _run_stage(self, *args, **kwargs):
        # All I want is to print this  directly before training starts.
        # Is that so hard to do?
        self._on_before_run()
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
        print(f'self.global_rank={self.global_rank}')
        if self.global_rank == 0:
            self._on_before_run_rank0()

    def _on_before_run_rank0(self):
        import rich
        dpath = self.log_dpath
        rich.print(f"Trainer log dpath:\n\n[link={dpath}]{dpath}[/link]\n")
