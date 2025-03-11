from root import ROOT_DIR  # Keep existing ROOT_DIR definition
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from lightning.pytorch import loggers
import argparse
from models import unet_precip_regression_lightning as unet_regr
from lightning.pytorch.tuner import Tuner

def train_regression(hparams, find_batch_size_automatically: bool = False):
    if hparams.model == "UNetDSAttention":
        net = unet_regr.UNetDSAttention(hparams=hparams)
    elif hparams.model == "UNetAttention":
        net = unet_regr.UNetAttention(hparams=hparams)
    elif hparams.model == "UNet":
        net = unet_regr.UNet(hparams=hparams)
    elif hparams.model == "UNetDS":
        net = unet_regr.UNetDS(hparams=hparams)
    else:
        raise NotImplementedError(f"Model '{hparams.model}' not implemented")

    # ========== HPC-SPECIFIC CHANGE ========== #
    default_save_path = "/scratch-shared/tmp.Udl4HYbZtd/data"
    # ========================================== #

    checkpoint_callback = ModelCheckpoint(
        dirpath=default_save_path / net.__class__.__name__,
        filename=net.__class__.__name__ + "_rain_threshold_50_{epoch}-{val_loss:.6f}",
        save_top_k=-1,
        verbose=False,
        monitor="val_loss",
        mode="min",
    )
    lr_monitor = LearningRateMonitor()
    tb_logger = loggers.TensorBoardLogger(save_dir=default_save_path, name=net.__class__.__name__)

    earlystopping_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=hparams.es_patience,
    )

    # ========== HPC OPTIMIZATION ========== #
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="16-mixed",  # Enable mixed precision for faster training
        fast_dev_run=hparams.fast_dev_run,
        max_epochs=hparams.epochs,
        default_root_dir=default_save_path,
        logger=tb_logger,
        callbacks=[checkpoint_callback, earlystopping_callback, lr_monitor],
        val_check_interval=hparams.val_check_interval,
    )
    # ====================================== #

    if find_batch_size_automatically:
        tuner = Tuner(trainer)
        tuner.scale_batch_size(net, mode="binsearch")

    trainer.fit(model=net, ckpt_path=hparams.resume_from_checkpoint)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = unet_regr.PrecipRegressionBase.add_model_specific_args(parser)

    # ========== HPC-SPECIFIC CHANGE ========== #
    parser.add_argument(
        "--dataset_folder",
        default=ROOT_DIR / "Data" / "Precipitation" / "era5_rain-threshold-train-test-v2-20.h5",
        type=str,
    )
    # ========================================== #

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--fast_dev_run", type=bool, default=False)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--val_check_interval", type=float, default=None)

    args = parser.parse_args()

    # ========== CRITICAL MISSING LINES ========== #
    args.n_channels = 12
    args.lr_patience = 4       # Required for learning rate scheduler
    args.es_patience = 15      # Required for EarlyStopping callback
    args.kernels_per_layer = 2
    args.use_oversampled_dataset = True
    # ============================================ #

    # ========== HPC-SPECIFIC DATA PATH ========== #
    args.dataset_folder = "/scratch-shared/tmp.Udl4HYbZtd/data"  # Hardcoded HPC path
    # ============================================ #

    print(f"Start training model: {args.model}")
    train_regression(args, find_batch_size_automatically=False)