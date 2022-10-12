import torch
from os.path import exists as file_exists
from wanda import config
from wanda.preprocessing.data_reader import HSDataReader
from wanda.data_loader.data_loader import create_hs_data_loader
from wanda.trainer.trainer import hs_model_trainer, sk_model_trainer
from wanda.infer.predict import Evaluator
from wanda.visualizer.visualize import Visualize
from wanda.model.cnn_hscore import WandaHSCNN
from wanda.model.svm import SVMModel
from wanda.model.isolation_forest import IsoForestModel
from wanda.model.lof import LOFModel
from wanda.preprocessing.preprocessor import HSCnnDataPreprocessor
from wanda.trainer.optimizer import optimize_iso_forest, optimize_svm


def main():
    if not file_exists(f"{config.BASE_PATH}/data/processed/train.csv"):
        data_reader = HSDataReader()
        data_reader.process_dataset()

    hs_train_loader = create_hs_data_loader(batch_size=config.BATCH_SIZE)
    hs_trainer = hs_model_trainer(hs_train_loader, progress_bar_refresh_rate=10)

    if not file_exists(f"{config.BASE_PATH}/models/WandaHSCNN.pt"):
        print("Train H-Score CNN First to train/predict SVM")
        return

    one_class_model_train()
    evaluate_models()
    visualize_activation_maps()
    optimize_hyperparameters()


def one_class_model_train():
    hs_train_loader = create_hs_data_loader(batch_size=config.BATCH_SIZE)
    hs_cnn_preprocessor = HSCnnDataPreprocessor()
    preprocessed_data, _ = hs_cnn_preprocessor.get_preprocess_data(hs_train_loader)

    svm_model = SVMModel()
    svm_model = sk_model_trainer(model=svm_model, preprocessed_data=preprocessed_data)

    iso_forest_model = IsoForestModel()
    iso_forest_model = sk_model_trainer(
        model=iso_forest_model, preprocessed_data=preprocessed_data
    )

    # lof_model = LOFModel()
    # lof_model = sk_model_trainer(model=lof_model, preprocessed_data=preprocessed_data)


def evaluate_models():
    svm_model = SVMModel()
    model_evaluator = Evaluator(model=svm_model)
    model_evaluator.evaulate()

    iso_f_model = IsoForestModel()
    model_evaluator = Evaluator(model=iso_f_model)
    model_evaluator.evaulate()

    # lof_model = LOFModel()
    # model_evaluator = Evaluator(model=lof_model)
    # model_evaluator.evaulate()


def visualize_activation_maps():
    cnn_hs = WandaHSCNN()
    cnn_hs.load_state_dict(torch.load(f"{config.BASE_PATH}/models/WandaHSCNN.pt"))
    cnn_hs.eval()

    visualizer = Visualize(hs_cnn=cnn_hs, train=False, n_samples=3)
    visualizer.random_visualize(labels=[1, 0])


def optimize_hyperparameters():
    hs_test_loader = create_hs_data_loader(
        batch_size=config.TEST_BATCH_SIZE, train=False, shuffle=True
    )
    hs_cnn_preprocessor = HSCnnDataPreprocessor()
    transformed_X, labels = hs_cnn_preprocessor.get_preprocess_data(hs_test_loader)
    if torch.is_tensor(transformed_X):
        transformed_X = transformed_X.detach().numpy()

    if torch.is_tensor(labels):
        labels = labels.detach().numpy()

    best_study_iso = optimize_iso_forest(transformed_X, labels, n_trials=100)

    print(
        f"Isolation Forest Best Params: {best_study_iso.best_params}. Best Value: {-1*best_study_iso.best_value}"
    )

    best_study_svm = optimize_svm(transformed_X, labels, n_trials=100)

    print(
        f"SVM Best Params: {best_study_svm.best_params}. Best Value: {-1*best_study_svm.best_value}"
    )


if __name__ == "__main__":
    main()
