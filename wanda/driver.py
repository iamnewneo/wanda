from os.path import exists as file_exists
from wanda import config
from wanda.preprocessing.data_reader import HSDataReader
from wanda.data_loader.data_loader import create_hs_data_loader
from wanda.trainer.trainer import hs_model_trainer, svm_trainer
from wanda.infer.predict import Evaluator


def main():
    if not file_exists(f"{config.BASE_PATH}/data/processed/train.csv"):
        data_reader = HSDataReader()
        data_reader.process_dataset()

    hs_train_loader = create_hs_data_loader(batch_size=config.BATCH_SIZE)
    hs_trainer = hs_model_trainer(hs_train_loader, progress_bar_refresh_rate=None)
    hs_model = hs_trainer.get_model()

    if not file_exists(f"{config.BASE_PATH}/models/WandaHSCNN.pt"):
        print("Train H-Score CNN First to train/predict SVM")
        return

    svm_model = svm_trainer(hs_train_loader)
    model_evaluator = Evaluator()
    model_evaluator.evaulate()


if __name__ == "__main__":
    main()
