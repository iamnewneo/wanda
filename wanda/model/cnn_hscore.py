import torch
from torch import nn
import numpy as np
from wanda import config
import pytorch_lightning as pl


class WandaHSCNN(pl.LightningModule):
    def __init__(self):
        super(WandaHSCNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=10, stride=1)
        self.relu1 = nn.ReLU()

        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=10)

        # self.cnn2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=10, stride=1)
        # self.relu2 = nn.ReLU()

        # self.maxpool2 = nn.MaxPool2d(kernel_size=10)

        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(3528, 15)
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        # out = self.cnn2(out)
        # out = self.relu2(out)
        # out = self.maxpool2(out)
        activation_maps = out
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        final_out = self.sigmoid1(out)
        return activation_maps, final_out

    def h_score(self, fx, gy):
        fx = fx - fx.mean(0)
        gy = gy - gy.mean(0)
        Nsamples = fx.size(0)
        covf = torch.matmul(fx.t(), fx) / Nsamples
        covg = torch.matmul(gy.t(), gy) / Nsamples
        h = -2 * torch.mean((fx * gy).sum(1)) + (covf * covg).sum()
        return h

    def loss_fn(self, X_1, X_2):
        return self.h_score(X_1, X_2)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=config.LR)
        return optimizer

    def training_step(self, batch, batch_idx):
        X_1 = batch["X_1"]
        X_2 = batch["X_2"]
        X_1_out = self(X_1)
        X_2_out = self(X_2)
        loss = self.loss_fn(X_1_out[1], X_2_out[1])
        return {"loss": loss}

    def training_epoch_end(self, train_step_outputs):
        avg_train_loss = torch.tensor([x["loss"] for x in train_step_outputs]).mean()
        self.temp_train_loss = avg_train_loss
        print(f"\nEpoch: {self.current_epoch} Train Loss: {self.temp_train_loss:.2f}")
        return {"loss": avg_train_loss}


class HSCnnDataPreprocessor:
    def __init__(self) -> None:
        self.cnn_hs = WandaHSCNN()
        self.cnn_hs.load_state_dict(
            torch.load(f"{config.BASE_PATH}/models/WandaHSCNN.pt")
        )
        self.cnn_hs.eval()

    def get_preprocess_data(self, data_loader):
        tranformed_images = []
        labels = []
        with torch.no_grad():
            for idx, batch in enumerate(data_loader):
                X_1_images = batch["X_1"]
                activation_maps, _ = self.cnn_hs(X_1_images)
                activation_maps = torch.flatten(activation_maps, start_dim=1)
                tranformed_images.append(activation_maps)
                labels.append(batch["label"].numpy())

        flattened_tranformed_images = torch.cat(tranformed_images)
        labels = np.concatenate(labels, axis=0)
        labels = labels.ravel()
        return flattened_tranformed_images, labels
