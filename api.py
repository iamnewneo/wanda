import torch
from PIL import Image
from sklearn.preprocessing import StandardScaler
from wanda import config
from wanda.model.cnn_hscore import WandaHSCNN
from wanda.model.svm import SVMModel
from wanda.model.isolation_forest import IsoForestModel
from wanda.utils.util import get_tranforms, load_object


def load_image(img_path):
    pytorch_tranforms = get_tranforms()
    X = Image.open(img_path).convert("RGB")
    X = pytorch_tranforms(X)
    X = X.type(torch.float32) / 255
    return X


class Predictor:
    def __init__(self, anomaly_threshold=0, anomaly_model_name="SVM") -> None:
        self.anomaly_threshold = anomaly_threshold
        self.anomaly_scaler = self.load_anomaly_scaler()
        self.hs_model = self.load_hscore_model()
        self.anomaly_model = self.load_anomaly_model(model_name=anomaly_model_name)

    def load_hscore_model(self):
        cnn_hs = WandaHSCNN()
        cnn_hs.load_state_dict(torch.load(f"{config.BASE_PATH}/models/WandaHSCNN.pt"))
        cnn_hs.eval()
        return cnn_hs

    def load_anomaly_model(self, model_name):
        clf = None
        if model_name == "SVM":
            clf = load_object(SVMModel.model_path)
        return clf

    def load_anomaly_scaler(self):
        scaler_path = f"{config.BASE_PATH}/models/HS_Scaler.pkl"
        return load_object(scaler_path)

    def is_anomaly(self, spectrogram):
        if spectrogram.dim() == 3:
            spectrogram = spectrogram.unsqueeze(dim=0)
        activation_maps, _ = self.hs_model(spectrogram)
        activation_maps = torch.flatten(activation_maps, start_dim=1)
        activation_maps = activation_maps.detach().numpy()

        activation_maps = self.anomaly_scaler.transform(activation_maps)
        anomaly_output = self.anomaly_model.decision_function(activation_maps)

        if len(anomaly_output) == 1:
            anomaly_output = anomaly_output[0]

        if anomaly_output > self.anomaly_threshold:
            return False
        return True
