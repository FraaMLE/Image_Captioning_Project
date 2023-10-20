import spacy
import pandas as pd
from typing import List, Dict
import matplotlib.pyplot as plt

spacy_english = spacy.load("en_core_web_sm")


class Words_Handler:
    def __init__(self, min_frequency: int, captions_path: str):
        self.df = pd.read_csv(captions_path)
        self.captions = self.df["caption"].tolist()
        self.min_frequency = min_frequency
        self.i2s = None
        self.s2i = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "UNK": 3}

    @staticmethod
    def tokenizer_text(text: str) -> List[str]:
        """
        converts a text string into a list of tokens
        """
        assert isinstance(text, str), "Input 'text' must be a string."
        return [token.text.lower() for token in spacy_english.tokenizer(text)]

    def frequencies(self) -> Dict[str, int]:
        """
        Count the occurrences of words in a list of captions.
        """
        assert isinstance(self.captions, list), "Input 'caption' must be a list of strings."
        occurrences = {}
        for caption in self.captions:
            for word in self.tokenizer_text(caption):
                if word not in occurrences:
                    occurrences[word] = 1
                else:
                    occurrences[word] += 1
        assert len(occurrences) == len(set(occurrences.keys())), "The dictionary contains duplicates."
        return occurrences

    @staticmethod
    def dictionary_sorter(unordered_dict: Dict, descending_order=True) -> List:
        assert isinstance(unordered_dict, dict), "Input 'occurrences' must be a dictionary."
        return sorted(unordered_dict.items(), key=lambda x: x[1], reverse=descending_order)

    def plot_word_histogram(self) -> None:
        """
        Plot histograms for the 10 most frequent and 10 least frequent words.
        """
        occurrences = self.frequencies()
        sorted_words = self.dictionary_sorter(occurrences)
        # Get the 10 most frequent and 10 least frequent words
        most_frequent = sorted_words[:10]
        least_frequent = sorted_words[-10:]
        # Separate words and frequencies
        most_frequent_words, most_frequent_counts = zip(*most_frequent)
        least_frequent_words, least_frequent_counts = zip(*least_frequent)
        # Create subplots for the two histograms
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        # Plot the 10 most frequent words
        ax1.barh(most_frequent_words, most_frequent_counts, color='blue')
        ax1.set_title('10 Most Frequent Words')
        ax1.set_xlabel('Frequency')
        # Plot the 10 least frequent words
        ax2.barh(least_frequent_words, least_frequent_counts, color='red')
        ax2.set_title('10 Least Frequent Words')
        ax2.set_xlabel('Frequency')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def invert_dict(dictionary: Dict) -> Dict:
        """ It swaps the keys and values of the s2i """
        return {v: k for k, v in dictionary.items()}

    def look_up_dictionary_builder(self):
        idx = 4
        occurrences = self.frequencies()
        for word in occurrences.keys():
            if occurrences[word] > self.min_frequency:
                self.s2i[word] = idx
                idx += idx
        self.i2s = self.invert_dict(self.s2i)

    def text2numbers(self, text: str) -> List[int]:
        text2tokens = self.tokenizer_text(text)
        return [
            self.s2i[token] if token in self.s2i else self.s2i["UNK"]
            for token in text2tokens
        ]

    def numbers2text(self, list_of_numbers: List[int]) -> List[str]:
        return [
            self.i2s[token] if token in self.i2s else self.i2s[3]
            for token in list_of_numbers
        ]


if __name__ == "__main__":
    handler = Words_Handler(min_frequency=5,
                            captions_path=r"C:\Users\franv\Downloads\Images_Dataset\flickr8k\captions.txt")
    handler.look_up_dictionary_builder()
    print(len(handler.s2i))
    print(len(handler.i2s))
    num = handler.text2numbers(handler.captions[0])
    print(num)
    print(handler.numbers2text(num))
    print(handler.captions[0])
    handler.plot_word_histogram()

