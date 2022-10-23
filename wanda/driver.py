import torch
from os.path import exists as file_exists
from wanda import config
from wanda.preprocessing.data_reader import HSDataReader
from wanda.data_loader.data_loader import create_hs_data_loader
from wanda.trainer.trainer import hs_model_trainer, sk_model_trainer
from wanda.infer.predict import Evaluator
from wanda.visualizer.visualize import Visualize
from wanda.model.cnn_hscore import WandaHSCNN
from wanda.model.lof import LOFModel
from wanda.preprocessing.preprocessor import HSCnnDataPreprocessor, SkDataPreprocessor
from wanda.trainer.optimizer import (
    optimize_iso_forest,
    optimize_svm,
    optimize_svdd,
    optimize_ecod,
)
from wanda.model.od_algos import DeepSVDDModel, ECODModel
from wanda.model.isolation_forest import IsoForestModel
from wanda.model.svm import SVMModel
from wanda.model.decompose import DecomposeData
from wanda.utils.util import load_object, save_object, save_text_results
from wanda.analyzer.analyze import detail_analyze_model


def main():
    # train_h_score_cnn()
    # one_class_model_train()
    # visualize_activation_maps()

    # train_data_decomposer()

    test_loader = create_hs_data_loader(
        batch_size=config.TEST_BATCH_SIZE, train=False, shuffle=False, greyscale=False
    )
    hs_data_preprocessor = HSCnnDataPreprocessor(svd_tranformation=False)
    transformed_X_hs, labels_hs, ids_hs = hs_data_preprocessor.get_preprocess_data(
        data_loader=test_loader, ids=True
    )

    if torch.is_tensor(transformed_X_hs):
        transformed_X_hs = transformed_X_hs.detach().numpy()

    if torch.is_tensor(labels_hs):
        labels_hs = labels_hs.detach().numpy()

    if torch.is_tensor(ids_hs):
        ids_hs = ids_hs.detach().numpy()

    test_loader = create_hs_data_loader(
        batch_size=config.TEST_BATCH_SIZE, train=False, shuffle=False, greyscale=True
    )
    sk_data_preprocessor = SkDataPreprocessor(svd_tranformation=False)
    transformed_X_sk, labels_sk, ids_sk = sk_data_preprocessor.get_preprocess_data(
        data_loader=test_loader, ids=True
    )

    if torch.is_tensor(transformed_X_sk):
        transformed_X_sk = transformed_X_sk.detach().numpy()

    if torch.is_tensor(labels_sk):
        labels_sk = labels_sk.detach().numpy()

    if torch.is_tensor(ids_sk):
        ids_sk = ids_sk.detach().numpy()

    optimize_hyperparameters_hscore_input(transformed_X_hs, labels_hs, ids_hs)
    optimize_hyperparameters_plain_input(transformed_X_sk, labels_sk, ids_sk)

    evaluate_best_models_hscore_input(transformed_X_hs, labels_hs, ids_hs)
    evaluate_best_models_plain(transformed_X_sk, labels_sk, ids_sk)

    detail_analyze_model()


def train_h_score_cnn():
    if not file_exists(f"{config.BASE_PATH}/data/processed/train.csv"):
        data_reader = HSDataReader()
        data_reader.process_dataset()

    hs_train_loader = create_hs_data_loader(batch_size=config.BATCH_SIZE)
    hs_trainer = hs_model_trainer(hs_train_loader, progress_bar_refresh_rate=10)

    if not file_exists(f"{config.BASE_PATH}/models/WandaHSCNN.pt"):
        print("Train H-Score CNN First to train/predict SVM")
        return


def train_data_decomposer():
    test_loader = create_hs_data_loader(
        batch_size=config.TEST_BATCH_SIZE, train=False, shuffle=False, greyscale=False
    )
    hs_data_preprocessor = HSCnnDataPreprocessor(svd_tranformation=False)
    transformed_X, labels = hs_data_preprocessor.get_preprocess_data(
        data_loader=test_loader
    )

    clf = DecomposeData()
    clf.fit(transformed_X)
    save_object(clf, DecomposeData.model_path)

    test_loader = create_hs_data_loader(
        batch_size=config.TEST_BATCH_SIZE, train=False, shuffle=False, greyscale=True
    )

    sk_data_preprocessor = SkDataPreprocessor(svd_tranformation=False)
    transformed_X, labels = sk_data_preprocessor.get_preprocess_data(
        data_loader=test_loader
    )
    clf = DecomposeData()
    clf.fit(transformed_X)
    save_object(clf, DecomposeData.model_path.replace(".pkl", "_plain.pkl"))


# def one_class_model_train():
#     hs_train_loader = create_hs_data_loader(batch_size=config.BATCH_SIZE)
#     hs_cnn_preprocessor = HSCnnDataPreprocessor()
#     preprocessed_data, _ = hs_cnn_preprocessor.get_preprocess_data(hs_train_loader)

#     svm_model = SVMModel()
#     svm_model = sk_model_trainer(model=svm_model, preprocessed_data=preprocessed_data)

#     iso_forest_model = IsoForestModel()
#     iso_forest_model = sk_model_trainer(
#         model=iso_forest_model, preprocessed_data=preprocessed_data
#     )

#     # lof_model = LOFModel()
#     # lof_model = sk_model_trainer(model=lof_model, preprocessed_data=preprocessed_data)


def evaluate_best_models_hscore_input(transformed_X, labels, ids):
    print("*" * 100)
    print("Evaluating Models Performance With H-Score Input")
    print("*" * 100)

    clf = load_object(SVMModel.model_path)
    model_evaluator = Evaluator(model=clf)
    model_evaluator.evaulate(transformed_X, labels, ids)

    clf = load_object(IsoForestModel.model_path)
    model_evaluator = Evaluator(model=clf)
    model_evaluator.evaulate(transformed_X, labels, ids)

    clf = load_object(ECODModel.model_path)
    model_evaluator = Evaluator(model=clf)
    model_evaluator.evaulate(transformed_X, labels, ids)

    clf = DeepSVDDModel(contamination=0.23781187437103915)
    clf.fit(transformed_X)
    model_evaluator = Evaluator(model=clf)
    model_evaluator.evaulate(transformed_X, labels, ids)


def evaluate_best_models_plain(transformed_X, labels, ids):
    print("*" * 100)
    print("Evaluating Models Performance Without H-Score Input")
    print("*" * 100)

    clf = load_object(SVMModel.model_path.replace(".pkl", "_plain.pkl"))
    model_evaluator = Evaluator(model=clf)
    model_evaluator.evaulate(transformed_X, labels, ids, save_postfix="plain")

    clf = load_object(IsoForestModel.model_path.replace(".pkl", "_plain.pkl"))
    model_evaluator = Evaluator(model=clf)
    model_evaluator.evaulate(transformed_X, labels, ids, save_postfix="plain")

    clf = load_object(ECODModel.model_path.replace(".pkl", "_plain.pkl"))
    model_evaluator = Evaluator(model=clf)
    model_evaluator.evaulate(transformed_X, labels, ids, save_postfix="plain")

    clf = DeepSVDDModel(contamination=0.43351217124174224)
    clf.fit(transformed_X)
    model_evaluator = Evaluator(model=clf)
    model_evaluator.evaulate(transformed_X, labels, ids, save_postfix="plain")


def visualize_activation_maps():
    cnn_hs = WandaHSCNN()
    cnn_hs.load_state_dict(torch.load(f"{config.BASE_PATH}/models/WandaHSCNN.pt"))
    cnn_hs.eval()

    visualizer = Visualize(hs_cnn=cnn_hs, train=False, n_samples=5)
    visualizer.random_visualize(labels=[1, 0])


def optimize_hyperparameters_hscore_input(transformed_X, labels, ids):
    print("*" * 100)
    print("Models Performance with H-Score Input")
    print("*" * 100)

    best_study_iso = optimize_iso_forest(
        transformed_X, labels, n_trials=config.N_OPT_TRIALS
    )
    best_model = best_study_iso.__getattribute__("_best_model")
    save_object(best_model, IsoForestModel.model_path)
    print(
        f"Isolation Forest Best Params: {best_study_iso.best_params}. Best Value: {-1*best_study_iso.best_value}"
    )
    best_study_svm = optimize_svm(transformed_X, labels, n_trials=config.N_OPT_TRIALS)
    best_model = best_study_svm.__getattribute__("_best_model")
    save_object(best_model, SVMModel.model_path)
    print(
        f"SVM Best Params: {best_study_svm.best_params}. Best Value: {-1*best_study_svm.best_value}"
    )
    best_study_ecod = optimize_ecod(transformed_X, labels, n_trials=config.N_OPT_TRIALS)
    best_model = best_study_ecod.__getattribute__("_best_model")
    save_object(best_model, ECODModel.model_path)
    print(
        f"ECOD Best Params: {best_study_ecod.best_params}. Best Value: {-1*best_study_ecod.best_value}"
    )
    n_trials = 5
    if config.ENV == "dev":
        n_trials = 1
    best_study_svdd = optimize_svdd(transformed_X, labels, n_trials=n_trials)
    best_model = best_study_svdd.__getattribute__("_best_model")
    # save_object(best_model, DeepSVDDModel.model_path)
    print(
        f"Deep SVDD Best Params: {best_study_svdd.best_params}. Best Value: {-1*best_study_svdd.best_value}"
    )

    print("*" * 100)
    print("*" * 100)
    result_string = f"""Summary: Models Performance with H-Score Input
    Isolation Forest Best Params: {best_study_iso.best_params}. Best Value: {-1*best_study_iso.best_value}
    SVM Best Params: {best_study_svm.best_params}. Best Value: {-1*best_study_svm.best_value}
    ECOD Best Params: {best_study_ecod.best_params}. Best Value: {-1*best_study_ecod.best_value}
    Deep SVDD Best Params: {best_study_svdd.best_params}. Best Value: {-1*best_study_svdd.best_value}
    """
    print(result_string)
    save_text_results(result_string, path=f"{config.BASE_PATH}/data/wih_HS_summary.txt")
    print("*" * 100)
    print("*" * 100)


def optimize_hyperparameters_plain_input(transformed_X, labels, ids):
    print("*" * 100)
    print("Models Performance without H-Score Input/Plain Input")
    print("*" * 100)
    best_study_iso = optimize_iso_forest(
        transformed_X, labels, n_trials=config.N_OPT_TRIALS
    )
    best_model = best_study_iso.__getattribute__("_best_model")
    save_object(
        best_model, IsoForestModel.model_path.replace(".pkl", "_plain.pkl"),
    )
    print(
        f"Isolation Forest Best Params: {best_study_iso.best_params}. Best Value: {-1*best_study_iso.best_value}"
    )
    best_study_svm = optimize_svm(transformed_X, labels, n_trials=config.N_OPT_TRIALS)
    best_model = best_study_svm.__getattribute__("_best_model")
    save_object(best_model, SVMModel.model_path.replace(".pkl", "_plain.pkl"))
    print(
        f"SVM Best Params: {best_study_svm.best_params}. Best Value: {-1*best_study_svm.best_value}"
    )
    best_study_ecod = optimize_ecod(transformed_X, labels, n_trials=config.N_OPT_TRIALS)
    best_model = best_study_ecod.__getattribute__("_best_model")
    save_object(best_model, ECODModel.model_path.replace(".pkl", "_plain.pkl"))
    print(
        f"ECOD Best Params: {best_study_ecod.best_params}. Best Value: {-1*best_study_ecod.best_value}"
    )
    n_trials = 5
    if config.ENV == "dev":
        n_trials = 1
    best_study_svdd = optimize_svdd(transformed_X, labels, n_trials=n_trials)
    best_model = best_study_svdd.__getattribute__("_best_model")
    # save_object(best_model, DeepSVDDModel.model_path.replace(".pkl", "_plain.pkl"))
    print(
        f"Deep SVDD Best Params: {best_study_svdd.best_params}. Best Value: {-1*best_study_svdd.best_value}"
    )
    print("*" * 100)
    print("*" * 100)
    result_string = f"""Summary: Models Performance without H-Score Input/Plain Input
    Isolation Forest Best Params: {best_study_iso.best_params}. Best Value: {-1*best_study_iso.best_value}
    SVM Best Params: {best_study_svm.best_params}. Best Value: {-1*best_study_svm.best_value}
    ECOD Best Params: {best_study_ecod.best_params}. Best Value: {-1*best_study_ecod.best_value}
    Deep SVDD Best Params: {best_study_svdd.best_params}. Best Value: {-1*best_study_svdd.best_value}
    """
    print(result_string)
    save_text_results(
        result_string, path=f"{config.BASE_PATH}/data/without_HS_summary.txt"
    )
    print("*" * 100)
    print("*" * 100)


if __name__ == "__main__":
    main()
