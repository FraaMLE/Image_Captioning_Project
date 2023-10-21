from data_handler.words_handler import Words_Handler
import unittest
import pandas as pd
import unittest
path = r"C:\Users\franv\Downloads\Images_Dataset\flickr8k\captions.txt"
dataframe = pd.read_csv(path)


def count_keys_greater_than_5(my_dict):
    count = 0
    for value in my_dict.values():
        if value > 5:
            count += 1
    return count


class TestWordsHandler(unittest.TestCase):
    def setUp(self):
        # You can initialize the Words_Handler instance in the setUp method
        self.handler = Words_Handler(min_frequency=5, captions_path=path)
        self.handler.look_up_dictionary_builder()

    def test_tokenizer_text(self):
        text = "This is a test sentence."
        tokens = self.handler.tokenizer_text(text)
        self.assertEqual(tokens, ['this', 'is', 'a', 'test', 'sentence', '.'])

    def test_frequencies(self):
        # Test the frequencies method
        occurrences = self.handler.frequencies()
        self.assertIsInstance(occurrences, dict)
        self.assertEqual(occurrences['swatting'], 1)  # Assuming 'test' appears once in test_captions.txt

    def test_dictionary_sorter(self):
        # Test the dictionary_sorter method
        test_dict = {'apple': 3, 'banana': 2, 'cherry': 1}
        sorted_dict = self.handler.dictionary_sorter(test_dict)
        self.assertEqual(sorted_dict, [('apple', 3), ('banana', 2), ('cherry', 1)])

    def test_look_up_dictionary_builder(self):
        """ Since {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "UNK": 3}, We need to add 4"""
        # Test the look_up_dictionary_builder method
        oc = self.handler.frequencies()
        self.assertIsNotNone(self.handler.i2s)
        self.assertEqual(len(self.handler.s2i), len(self.handler.i2s))
        self.assertGreater(len(self.handler.s2i), 4)  # Check that the dictionary has grown
        self.assertEqual(len(self.handler.s2i), count_keys_greater_than_5(oc)+4)

    def test_text2numbers(self):
        # Test the text2numbers method
        text = "This is a test sentence ."
        numbers = self.handler.text2numbers(text)
        self.assertIsInstance(numbers, list)
        self.assertEqual(len(numbers), len(text.split()))  # Check the length of the output

    def test_numbers2text(self):
        # Test the numbers2text method
        numbers = [1, 2, 3, 4, 5, 0]
        text = self.handler.numbers2text(numbers)
        self.assertIsInstance(text, list)
        self.assertEqual(len(text), len(numbers))  # Check the length of the output


if __name__ == '__main__':
    unittest.main()
