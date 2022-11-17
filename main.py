import pandas as pd
from api import Predictor, load_image


## Globally Load once, so we dont have to load
## underlying models on every inference call
predictor = Predictor()


def is_anomaly(spectrogram):
    is_anomalous = predictor.is_anomaly(spectrogram)
    return is_anomalous


def main():
    img_path = "./data/Test_Data_Set_With_Interference/WiFi2_20/BLU5_0_3/intW2B5_20_0_3_17501_OFF.png"
    torch_image = load_image(img_path)
    is_anomalous = is_anomaly(torch_image)
    print(f"Is Anomaly? {is_anomalous}")


if __name__ == "__main__":
    main()
