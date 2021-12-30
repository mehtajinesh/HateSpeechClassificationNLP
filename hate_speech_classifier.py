"""This module is used for performing hate speech classification."""
from pandas import read_csv


class HateSpeechClassifier:
    """
    Hate Speech Classification Algorithm.
    """

    def __init__(self):
        """
        Initialize Hate Speech Classification.
        """
        self.dataframe = None

    def fetch_data(self, data_path: str):
        """
        Fetch data from the given file path and stores into dataframe.
        Note: the data file should be in csv format.
        """
        if data_path is None or data_path == '':
            raise ValueError("Data path is not specified.")
        self.dataframe = read_csv(data_path)

    def validate_fetched_data(self):
        """
        Validate fetched data.
        """
        if self.dataframe is None:
            raise ValueError("Data is not fetched.")
        print("Data Summary:")
        print(self.dataframe.info())
        print("Sample data:")
        print(self.dataframe.head())

    def preprocess_data(self):
        """
        Preprocess data.
        """
        pass

    def generate_features(self):
        """
        Generate features.
        """
        pass

    def train_model(self):
        """
        Train model.
        """
        pass

    def predict(self):
        """
        Predict.
        """
        pass

    def evaluate(self):
        """
        Evaluate.
        """
        pass

    def save_model(self):
        """
        Save model.
        """
        pass

    def load_model(self):
        """
        Load model.
        """
        pass
