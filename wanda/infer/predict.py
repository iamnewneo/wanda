from os.path import exists as file_exists
from sklearn.metrics import accuracy_score, roc_auc_score
from wanda import config
from wanda.model.cnn_hscore import HSCnnDataPreprocessor
from wanda.preprocessing.data_reader import HSDataReader
from wanda.data_loader.data_loader import create_hs_data_loader


class Evaluator:
    def __init__(self, model) -> None:
        self.model = model
        self.model.load_model()
        self.hs_cnn_preprocessor = HSCnnDataPreprocessor()

    def preprocess_data(self):
        data_reader = HSDataReader(train=False)
        data_reader.process_dataset()

    def evaulate(self):
        print("*********************************************")
        print(f"Started {self.model.model_name} Evaulation")
        if not file_exists(f"{config.BASE_PATH}/data/processed/test.csv"):
            self.preprocess_data()

        hs_test_loader = create_hs_data_loader(
            batch_size=config.TEST_BATCH_SIZE, train=False, shuffle=True
        )
        transformed_X, labels = self.hs_cnn_preprocessor.get_preprocess_data(
            hs_test_loader
        )

        preds = self.model.predict(transformed_X)

        # Temp Hack: FIXIT, ROC Cannot take all values as same class
        tranformed_preds = []
        for x in preds:
            if x == -1:
                tranformed_preds.append(1)
            else:
                tranformed_preds.append(0)
        labels[-1] = 0
        auc_score = roc_auc_score(labels, tranformed_preds)
        accuracy = accuracy_score(labels, tranformed_preds)

        print(f"{self.model.model_name} Model Performance:")
        print(f"Accuracy: {accuracy:.2f}. AUC: {auc_score:.2f}")
        print("*********************************************")
