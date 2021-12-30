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


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        print(error)
