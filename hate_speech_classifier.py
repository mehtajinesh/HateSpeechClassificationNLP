"""This module is used for performing hate speech classification."""
from pandas import read_csv
from preprocessing import PreprocessingData
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score


class HateSpeechClassifier:
    """Hate Speech Classification Algorithm.
    """

    def __init__(self):
        """
        Initialize Hate Speech Classification.
        """
        self.dataframe = None
        self.text_column = ""
        self.label_column = ""
        self.tfidf_vectorizer = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    def fetch_data(self, data_path: str):
        """
        Fetch data from the given file path and stores into dataframe.
        Note: the data file should be in csv format.
        """
        if data_path is None or data_path == '':
            raise ValueError("Data path is not specified.")
        self.dataframe = read_csv(data_path)
        self.text_column = "tweet"
        self.label_column = "class"

    def validate_loaded_data(self):
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
        preprocess = PreprocessingData()
        text_column_dataframe = self.dataframe[self.text_column]
        text_column_dataframe = preprocess.perform_lower_casing(
            text_column_dataframe)
        text_column_dataframe = preprocess.remove_extra_whitespaces(
            text_column_dataframe)
        text_column_dataframe = preprocess.perform_tokenization(
            text_column_dataframe)
        text_column_dataframe = preprocess.remove_stopwords(
            text_column_dataframe)
        text_column_dataframe = preprocess.remove_punctuations(
            text_column_dataframe)
        text_column_dataframe = preprocess.perform_lemmatize(
            text_column_dataframe)
        text_column_dataframe = preprocess.perform_stemming(
            text_column_dataframe)
        self.dataframe[self.text_column] = text_column_dataframe

    def split_data(self):
        """
        Split data.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.dataframe[self.text_column], self.dataframe[self.label_column], test_size=0.33)

    def dummy_fun(self, doc):
        return doc

    def generate_features(self):
        """
        Generate features from data.
        """
        self.tfidf_vectorizer = TfidfVectorizer(
            tokenizer=self.dummy_fun, preprocessor=self.dummy_fun)
        self.X_train = self.tfidf_vectorizer.fit_transform(self.X_train)

    def train_model(self):
        """
        Train model.
        """
        self.model = MultinomialNB().fit(self.X_train, self.y_train)

    def predict(self, text: str):
        """
        Predict.
        """
        vectorized_text = self.tfidf_vectorizer.fit_transform(text)
        return self.model.predict(vectorized_text)

    def evaluate_model(self):
        """
        Evaluate.
        """
        self.X_test = self.tfidf_vectorizer.transform(self.X_test)
        accuracy = self.model.score(self.X_test, self.y_test)
        return accuracy

    def f1_score(self):
        """
        F1 score.
        """
        predictions = self.model.predict(self.X_test)
        return f1_score(self.y_test, predictions, average=None)

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
