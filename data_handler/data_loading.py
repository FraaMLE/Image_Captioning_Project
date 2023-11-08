import os
import pandas as pd
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
from data_handler.words_handler import Words_Handler
import matplotlib.pyplot as plt


class Data_set(Dataset):
    def __init__(self, parent_dir_path: str, captions_files_path: str, transform=None):
        self.parent_dir = parent_dir_path
        self.df = pd.read_csv(captions_files_path)
        self.transform = transform
        self.imgs = self.df['image']
        self.captions = self.df['caption']
        self.vocab = Words_Handler(min_frequency=5, captions_path=captions_files_path)
        self.i2s = self.vocab.i2s
        self.s2i = self.vocab.s2i

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.parent_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        numerical_list = [self.s2i["<SOS>"], self.s2i["<EOS>"]]
        numerical_list[1:1] = self.vocab.text2numbers(caption)

        return img, torch.tensor(numerical_list)


class Customized_collate:
    def __init__(self, pad_idx = 0):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = torch.cat([item[0].unsqueeze(0) for item in batch], dim=0)
        targets = pad_sequence([item[1] for item in batch],
                               batch_first=True, padding_value=self.pad_idx)
        return imgs, targets


class ConcatenateFeaturesAndSequences(torch.nn.Module):
    def __init__(self, embed_size):
        super(ConcatenateFeaturesAndSequences, self).__init__()
        self.embed_size = embed_size

    def forward(self, features, embedded_captions):
        """ Expand the feature tensor to have the same shape as sequence tensors:
         unsqueeze ===> (N,embed_size) ---> (N,1,embed_size)
         expand    ===> (N,1,embed_size) ---> (N,sequence_length,embed_size) """
        expanded_features = features.unsqueeze(1).expand(-1, embedded_captions.size(1), -1)

        """ Concatenate the feature tensor and the sequence tensor along the last dimension
         cat      ===> (N,sequence_length,embed_size) ---> (N,sequence_length,embed_size*2) """
        result = torch.cat((expanded_features, embedded_captions), dim=2)
        return result





if __name__ == "__main__":
    captions_files_path = r"C:\Users\franv\Downloads\Images_Dataset\flickr8k\captions.txt"
    parent_dir_path = r"C:\Users\franv\Downloads\Images_Dataset\flickr8k\images"
    # Step 1: Define image transforms (adjust to your needs)
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    custom_dataset = Data_set(parent_dir_path = parent_dir_path, captions_files_path = captions_files_path, transform=image_transform)
    batch_size = 32  # Adjust this according to your needs
    data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, collate_fn=Customized_collate())
    for images, captions in data_loader:
        print(captions.shape)
        print(images.shape)
        print(captions[0][:])
        print(captions[0].shape)
        break
