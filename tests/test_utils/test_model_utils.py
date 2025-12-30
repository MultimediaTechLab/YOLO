import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest
import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from yolo.config.config import DataConfig, SchedulerConfig
from yolo.utils.model_utils import GradientAccumulation, lerp


class TestLerp:
    """Test the lerp (linear interpolation) function."""

    def test_lerp_basic(self):
        """Test basic linear interpolation."""
        assert lerp(0, 10, 0, 10) == 0
        assert lerp(0, 10, 5, 10) == 5
        assert lerp(0, 10, 10, 10) == 10

    def test_lerp_fractional(self):
        """Test linear interpolation with fractional steps."""
        assert lerp(0, 100, 25, 100) == 25
        assert lerp(0, 100, 50, 100) == 50
        assert lerp(0, 100, 75, 100) == 75

    def test_lerp_negative_values(self):
        """Test linear interpolation with negative values."""
        assert lerp(-10, 10, 0, 10) == -10
        assert lerp(-10, 10, 5, 10) == 0
        assert lerp(-10, 10, 10, 10) == 10

    def test_lerp_reverse_range(self):
        """Test linear interpolation from larger to smaller value."""
        assert lerp(10, 0, 0, 10) == 10
        assert lerp(10, 0, 5, 10) == 5
        assert lerp(10, 0, 10, 10) == 0


class TestGradientAccumulation:
    """Test the GradientAccumulation callback."""

    @pytest.fixture
    def data_cfg(self):
        """Create a mock DataConfig for testing."""
        cfg = Mock(spec=DataConfig)
        cfg.equivalent_batch_size = 64
        cfg.batch_size = 16
        return cfg

    @pytest.fixture
    def scheduler_cfg_with_warmup(self):
        """Create a mock SchedulerConfig with warmup."""
        cfg = Mock(spec=SchedulerConfig)
        # Create a warmup object that supports attribute access
        warmup = Mock()
        warmup.epochs = 3
        cfg.warmup = warmup
        return cfg

    @pytest.fixture
    def scheduler_cfg_without_warmup(self):
        """Create a mock SchedulerConfig without warmup."""
        cfg = Mock(spec=SchedulerConfig)
        # Create a warmup object without epochs attribute
        warmup = Mock(spec=[])  # Empty spec means no attributes
        cfg.warmup = warmup
        return cfg

    @pytest.fixture
    def mock_trainer(self):
        """Create a mock Trainer."""
        trainer = Mock()
        trainer.world_size = 1
        trainer.global_step = 0
        trainer.accumulate_grad_batches = 1
        return trainer

    @pytest.fixture
    def mock_pl_module(self):
        """Create a mock LightningModule."""
        pl_module = Mock()
        # Mock train_loader with 100 batches
        pl_module.train_loader = list(range(100))
        return pl_module

    def test_init_with_warmup(self, data_cfg, scheduler_cfg_with_warmup):
        """Test initialization with warmup configuration."""
        callback = GradientAccumulation(data_cfg, scheduler_cfg_with_warmup)
        
        assert callback.equivalent_batch_size == 64
        assert callback.actual_batch_size == 16
        assert callback.warmup_epochs == 3
        assert callback.current_batch == 0
        assert callback.max_accumulation == 1
        assert callback.warmup_batches == 0

    def test_init_without_warmup(self, data_cfg, scheduler_cfg_without_warmup):
        """Test initialization without warmup configuration."""
        callback = GradientAccumulation(data_cfg, scheduler_cfg_without_warmup)
        
        assert callback.equivalent_batch_size == 64
        assert callback.actual_batch_size == 16
        assert callback.warmup_epochs == 0
        assert callback.current_batch == 0
        assert callback.max_accumulation == 1
        assert callback.warmup_batches == 0

    def test_setup_single_gpu(self, data_cfg, scheduler_cfg_with_warmup, mock_trainer, mock_pl_module):
        """Test setup method with single GPU (world_size=1)."""
        callback = GradientAccumulation(data_cfg, scheduler_cfg_with_warmup)
        mock_trainer.world_size = 1
        
        callback.setup(mock_trainer, mock_pl_module, "fit")
        
        # equivalent_batch_size=64, actual_batch_size=16, world_size=1
        # effective_batch_size = 16 * 1 = 16
        # max_accumulation = round(64 / 16) = 4
        assert callback.max_accumulation == 4
        # warmup_batches = warmup_epochs * batches_per_epoch
        # warmup_batches = 3 * (100 / 1) = 300
        assert callback.warmup_batches == 300

    def test_setup_multi_gpu(self, data_cfg, scheduler_cfg_with_warmup, mock_trainer, mock_pl_module):
        """Test setup method with multiple GPUs (world_size=4)."""
        callback = GradientAccumulation(data_cfg, scheduler_cfg_with_warmup)
        mock_trainer.world_size = 4
        
        callback.setup(mock_trainer, mock_pl_module, "fit")
        
        # equivalent_batch_size=64, actual_batch_size=16, world_size=4
        # effective_batch_size = 16 * 4 = 64
        # max_accumulation = round(64 / 64) = 1
        assert callback.max_accumulation == 1
        # warmup_batches = warmup_epochs * batches_per_epoch
        # warmup_batches = 3 * (100 / 4) = 75
        assert callback.warmup_batches == 75

    def test_setup_fractional_accumulation(self, scheduler_cfg_with_warmup, mock_trainer, mock_pl_module):
        """Test setup with fractional accumulation (should round to nearest int)."""
        data_cfg = Mock(spec=DataConfig)
        data_cfg.equivalent_batch_size = 100
        data_cfg.batch_size = 16
        
        callback = GradientAccumulation(data_cfg, scheduler_cfg_with_warmup)
        mock_trainer.world_size = 1
        
        callback.setup(mock_trainer, mock_pl_module, "fit")
        
        # equivalent_batch_size=100, actual_batch_size=16, world_size=1
        # effective_batch_size = 16 * 1 = 16
        # max_accumulation = round(100 / 16) = round(6.25) = 6
        assert callback.max_accumulation == 6

    def test_setup_minimum_accumulation(self, scheduler_cfg_with_warmup, mock_trainer, mock_pl_module):
        """Test that max_accumulation is at least 1."""
        data_cfg = Mock(spec=DataConfig)
        data_cfg.equivalent_batch_size = 16
        data_cfg.batch_size = 32
        
        callback = GradientAccumulation(data_cfg, scheduler_cfg_with_warmup)
        mock_trainer.world_size = 2
        
        callback.setup(mock_trainer, mock_pl_module, "fit")
        
        # equivalent_batch_size=16, actual_batch_size=32, world_size=2
        # effective_batch_size = 32 * 2 = 64
        # max_accumulation = max(1, round(16 / 64)) = max(1, 0) = 1
        assert callback.max_accumulation == 1

    def test_on_train_epoch_start(self, data_cfg, scheduler_cfg_with_warmup, mock_trainer, mock_pl_module):
        """Test that current_batch is updated at epoch start."""
        callback = GradientAccumulation(data_cfg, scheduler_cfg_with_warmup)
        mock_trainer.global_step = 150
        
        callback.on_train_epoch_start(mock_trainer, mock_pl_module)
        
        assert callback.current_batch == 150

    def test_on_train_batch_start_before_warmup(self, data_cfg, scheduler_cfg_with_warmup, mock_trainer, mock_pl_module):
        """Test gradient accumulation during warmup phase."""
        callback = GradientAccumulation(data_cfg, scheduler_cfg_with_warmup)
        callback.setup(mock_trainer, mock_pl_module, "fit")
        
        # warmup_batches = 300, max_accumulation = 4
        assert callback.warmup_batches == 300
        assert callback.max_accumulation == 4
        
        # At batch 0 (start of warmup), should be 1
        callback.current_batch = 0
        callback.on_train_batch_start(mock_trainer, mock_pl_module)
        assert mock_trainer.accumulate_grad_batches == 1
        
        # At batch 75 (25% through warmup), should interpolate
        # lerp(1, 4, 75, 300) = 1 + (4-1) * 75/300 = 1 + 0.75 = 1.75 -> round to 2
        callback.current_batch = 75
        callback.on_train_batch_start(mock_trainer, mock_pl_module)
        assert mock_trainer.accumulate_grad_batches == 2
        
        # At batch 150 (50% through warmup), should be halfway
        # lerp(1, 4, 150, 300) = 1 + (4-1) * 150/300 = 1 + 1.5 = 2.5 -> round to 2 or 3
        callback.current_batch = 150
        callback.on_train_batch_start(mock_trainer, mock_pl_module)
        assert mock_trainer.accumulate_grad_batches == round(1 + (4-1) * 150/300)
        
        # At batch 225 (75% through warmup)
        # lerp(1, 4, 225, 300) = 1 + (4-1) * 225/300 = 1 + 2.25 = 3.25 -> round to 3
        callback.current_batch = 225
        callback.on_train_batch_start(mock_trainer, mock_pl_module)
        assert mock_trainer.accumulate_grad_batches == 3

    def test_on_train_batch_start_after_warmup(self, data_cfg, scheduler_cfg_with_warmup, mock_trainer, mock_pl_module):
        """Test gradient accumulation after warmup phase."""
        callback = GradientAccumulation(data_cfg, scheduler_cfg_with_warmup)
        callback.setup(mock_trainer, mock_pl_module, "fit")
        
        # warmup_batches = 300, max_accumulation = 4
        assert callback.warmup_batches == 300
        assert callback.max_accumulation == 4
        
        # At batch 300 (end of warmup), should be max_accumulation
        callback.current_batch = 300
        callback.on_train_batch_start(mock_trainer, mock_pl_module)
        assert mock_trainer.accumulate_grad_batches == 4
        
        # At batch 500 (well after warmup), should still be max_accumulation
        callback.current_batch = 500
        callback.on_train_batch_start(mock_trainer, mock_pl_module)
        assert mock_trainer.accumulate_grad_batches == 4

    def test_on_train_batch_start_no_warmup(self, data_cfg, scheduler_cfg_without_warmup, mock_trainer, mock_pl_module):
        """Test gradient accumulation when warmup is disabled."""
        callback = GradientAccumulation(data_cfg, scheduler_cfg_without_warmup)
        callback.setup(mock_trainer, mock_pl_module, "fit")
        
        # warmup_batches = 0, max_accumulation = 4
        assert callback.warmup_batches == 0
        assert callback.max_accumulation == 4
        
        # From the start, should use max_accumulation
        callback.current_batch = 0
        callback.on_train_batch_start(mock_trainer, mock_pl_module)
        assert mock_trainer.accumulate_grad_batches == 4
        
        callback.current_batch = 100
        callback.on_train_batch_start(mock_trainer, mock_pl_module)
        assert mock_trainer.accumulate_grad_batches == 4

    def test_on_train_batch_end(self, data_cfg, scheduler_cfg_with_warmup, mock_trainer, mock_pl_module):
        """Test that current_batch is incremented after each batch."""
        callback = GradientAccumulation(data_cfg, scheduler_cfg_with_warmup)
        callback.current_batch = 0
        
        callback.on_train_batch_end(mock_trainer, mock_pl_module)
        assert callback.current_batch == 1
        
        callback.on_train_batch_end(mock_trainer, mock_pl_module)
        assert callback.current_batch == 2
        
        callback.on_train_batch_end(mock_trainer, mock_pl_module)
        assert callback.current_batch == 3

    def test_full_training_cycle(self, data_cfg, scheduler_cfg_with_warmup, mock_trainer, mock_pl_module):
        """Test a complete training cycle with warmup."""
        callback = GradientAccumulation(data_cfg, scheduler_cfg_with_warmup)
        callback.setup(mock_trainer, mock_pl_module, "fit")
        
        # Start of epoch
        callback.on_train_epoch_start(mock_trainer, mock_pl_module)
        assert callback.current_batch == 0
        
        # First batch - should start at accumulation of 1
        callback.on_train_batch_start(mock_trainer, mock_pl_module)
        assert mock_trainer.accumulate_grad_batches == 1
        callback.on_train_batch_end(mock_trainer, mock_pl_module)
        assert callback.current_batch == 1
        
        # Simulate warmup progression
        for _ in range(149):  # Complete up to batch 150
            callback.on_train_batch_start(mock_trainer, mock_pl_module)
            callback.on_train_batch_end(mock_trainer, mock_pl_module)
        
        assert callback.current_batch == 150
        
        # Continue past warmup (warmup_batches = 300)
        for _ in range(200):  # Go to batch 350
            callback.on_train_batch_start(mock_trainer, mock_pl_module)
            callback.on_train_batch_end(mock_trainer, mock_pl_module)
        
        assert callback.current_batch == 350
        # After warmup, should be at max_accumulation
        callback.on_train_batch_start(mock_trainer, mock_pl_module)
        assert mock_trainer.accumulate_grad_batches == 4
