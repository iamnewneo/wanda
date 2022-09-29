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
    trainer.fit(model, train_dataloader)
    torch.save(model.state_dict(), f"{config.BASE_PATH}/models/WandaHSCNN.pt")
    print(f"H-Score CNN Model Saved at: {config.BASE_PATH}/models/WandaHSCNN.pt")
    return trainer


def sk_model_trainer(model, data_loader):
    print("*********************************************")
    print(f"Training: {model.model_name}")
    model.fit(data_loader)
    model.save_model()
    print("*********************************************")
    return model
