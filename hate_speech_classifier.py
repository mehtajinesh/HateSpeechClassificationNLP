"""This module is used for performing hate speech classification."""
from pandas import read_csv
from preprocessing import PreprocessingData
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel


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
        text_column_dataframe = preprocess.perform_url_mention_data_from_tweet(
            text_column_dataframe)
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
        pipe = Pipeline(
            [('select', SelectFromModel(LogisticRegression(class_weight='balanced',
                                                           penalty="l1", C=0.01, solver='liblinear'))),
             ('model', LogisticRegression(class_weight='balanced', penalty='l2'))])
        param_grid = [{}]  # Optionally add parameters here
        grid_search = GridSearchCV(pipe,
                                   param_grid,
                                   cv=StratifiedKFold(n_splits=5).split(
                                       self.X_train, self.y_train),
                                   verbose=2)
        self.model = grid_search.fit(self.X_train, self.y_train)

    def predict(self, text: str):
        """
        Predict.
        """
        vectorized_text = self.tfidf_vectorizer.transform([text])
        return self.model.predict(vectorized_text)

    def evaluate_model(self):
        """
        Evaluate.
        """
        transformed_test = self.tfidf_vectorizer.transform(self.X_test)
        accuracy = self.model.score(transformed_test, self.y_test)
        return accuracy

    def get_classification_report(self):
        """
        Generates classification report.
        """
        transformed_test = self.tfidf_vectorizer.transform(self.X_test)
        predictions = self.model.predict(transformed_test)
        return classification_report(self.y_test, predictions)

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
