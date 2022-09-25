from os.path import exists as file_exists
from sklearn.metrics import accuracy_score, roc_auc_score
from wanda import config
from wanda.model.svm import SVMModel
from wanda.preprocessing.data_reader import HSDataReader
from wanda.data_loader.data_loader import create_hs_data_loader


class Evaluator:
    def __init__(self) -> None:
        self.svm_model = SVMModel()
        self.svm_model.load_model()

    def preprocess_data(self):
        data_reader = HSDataReader(train=False)
        data_reader.process_dataset()

    def evaulate(self):
        print("Startign Evaulate")
        if not file_exists(f"{config.BASE_PATH}/data/processed/test.csv"):
            self.preprocess_data()

        hs_test_loader = create_hs_data_loader(
            batch_size=config.TEST_BATCH_SIZE, train=False, shuffle=True
        )
        print("Created Test Loader")
        transformed_X, labels = self.svm_model.get_preprocess_data(hs_test_loader)
        print("Used H-Score")
        preds = self.svm_model.predict(transformed_X)
        print("used SVM")

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

        print("Current Model Performance:")
        print(f"Accuracy: {accuracy:.2f}. AUC: {auc_score:.2f}")
