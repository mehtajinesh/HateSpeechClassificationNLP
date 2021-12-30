""" This module contains the main function."""
from hate_speech_classifier import HateSpeechClassifier


def main():
    """main function
    """
    hate_speech_classify = HateSpeechClassifier()
    hate_speech_classify.fetch_data('data/hate_speech_data.csv')
    hate_speech_classify.validate_fetched_data()


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        print(error)
