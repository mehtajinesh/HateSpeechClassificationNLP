""" This module contains the main function."""
from hate_speech_classifier import HateSpeechClassifier


def main():
    """main function
    """
    hate_speech_classify = HateSpeechClassifier()
    print("Fetching data...")
    hate_speech_classify.fetch_data('data/hate_speech_data.csv')
    hate_speech_classify.validate_loaded_data()
    print("Preprocessing data...")
    hate_speech_classify.preprocess_data()
    hate_speech_classify.validate_loaded_data()
    print("Splitting data...")
    hate_speech_classify.split_data()
    print("Generating Features...")
    hate_speech_classify.generate_features()
    print("Training model...")
    hate_speech_classify.train_model()
    print("Evaluting Model...")
    accuracy = hate_speech_classify.evaluate_model()
    print(f'Accuracy: {accuracy}')
    f1_score = hate_speech_classify.f1_score()
    print(f'F1 Score: {f1_score}')


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        print(error)
