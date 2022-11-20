from api import Predictor, load_image


## Globally Load once, so we dont have to load
## underlying models on every inference call
predictor = Predictor()


def is_anomaly(spectrogram):
    """
    Args:
        spectrogram (Tensor): Single Tensor image of size (C, H, W).
    Returns:
        Bool: True if spectrogram has an anomaly False otherwise
    """
    is_anomalous = predictor.is_anomaly(spectrogram)
    return is_anomalous


def test_anomly():
    img_path_anomaly = "./data/Test_Data_Set_With_Interference/WiFi2_20/BLU5_0_2/intW2B5_20_0_2_16311_ON.png"
    torch_image = load_image(img_path_anomaly)
    is_anomalous = is_anomaly(torch_image)
    print(f"Actual Anomaly True, Predicted: {is_anomalous}")


def test_non_anomaly():
    img_path_no_anomaly = "./data/Test_Data_Set_With_Interference/WiFi2_20/BLU5_0_3/intW2B5_20_0_3_17501_OFF.png"
    torch_image = load_image(img_path_no_anomaly)
    is_anomalous = is_anomaly(torch_image)
    print(f"Actual Anomaly False, Predicted: {is_anomalous}")


def main():
    test_anomly()
    test_non_anomaly()


if __name__ == "__main__":
    main()
