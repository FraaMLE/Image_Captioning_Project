import os
import unittest
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from data_handler.words_handler import Words_Handler  # Import your Words_Handler module

# Import the class to be tested
from data_handler.data_loading import Data_set, Customized_collate  # Replace 'your_module' with your actual module name

captions_files_path = r"C:\Users\franv\Downloads\Images_Dataset\flickr8k\captions.txt"
parent_dir_path = r"C:\Users\franv\Downloads\Images_Dataset\flickr8k\images"


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # Define some sample data for testing
        self.sample_data = {
            'image': ['image1.jpg', 'image2.jpg'],
            'caption': ['This is caption 1', 'This is caption 2']
        }
        # Create a sample DataFrame
        self.sample_df = pd.DataFrame(self.sample_data)

    def test_data_loading(self):
        # Create a temporary directory for testing
        test_dir = 'test_images'
        os.makedirs(test_dir, exist_ok=True)

        try:
            # Save sample images in the test directory
            for img_name in self.sample_data['image']:
                img = Image.new('RGB', (100, 100))
                img.save(os.path.join(test_dir, img_name))

            # Create a sample dataset
            image_transform = transforms.Compose([transforms.ToTensor()])
            dataset = Data_set(test_dir, self.sample_df, transform=image_transform)

            # Check if the dataset is of the correct length
            self.assertEqual(len(dataset), len(self.sample_df))

            # Check if the dataset correctly loads images and captions
            img, caption = dataset[0]
            self.assertIsInstance(img, torch.Tensor)
            self.assertIsInstance(caption, torch.Tensor)

            # Ensure Customized_collate works as expected
            collate_fn = Customized_collate()
            batch = [(img, caption), (img, caption)]
            batch_imgs, batch_captions = collate_fn(batch)
            self.assertIsInstance(batch_imgs, torch.Tensor)
            self.assertIsInstance(batch_captions, torch.Tensor)

        finally:
            # Clean up: remove the temporary directory and its contents
            for img_name in self.sample_data['image']:
                os.remove(os.path.join(test_dir, img_name))
            os.rmdir(test_dir)

if __name__ == '__main__':
    unittest.main()
