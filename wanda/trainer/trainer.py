import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from wanda import config
from wanda.model.cnn_hscore import WandaHSCNN
from wanda.model.svm import SVMModel


def hs_model_trainer(train_dataloader, progress_bar_refresh_rate):
    model = WandaHSCNN()
    gpus = None
    precision = 32
    if config.DEVICE in ["gpu", "cuda", "cuda:0"]:
        gpus = config.N_GPU
        precision = config.FP_PRECISION
    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=config.MAX_EPOCHS,
        min_epochs=1,
        weights_summary=None,
        progress_bar_refresh_rate=progress_bar_refresh_rate,
        precision=precision,
    )
    cnn_hs_model = trainer.get_model()
    torch.save(cnn_hs_model.state_dict(), f"{config.BASE_PATH}/models/WandaHSCNN.pt")
    trainer.fit(model, train_dataloader)
    return trainer


def svm_trainer():
    svm_model = SVMModel()
    svm_model.fit()


if __name__ == "__main__":
    # hs_model_trainer()
    svm_trainer()
