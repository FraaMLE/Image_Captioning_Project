import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from data_handler.data_loading import Data_set, Customized_collate
from model.lrcn import Image_captioner
from data_handler.words_handler import Words_Handler
from training.trainer import train

""" Set the Transformation to apply to the images """
image_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
])

""" Set the batch size and epochs """
batch_size = 32
epochs = 10

""" Initialize the Dataloader """
captions_files_path = r"C:\Users\franv\Downloads\Images_Dataset\flickr8k\captions.txt"
parent_dir_path = r"C:\Users\franv\Downloads\Images_Dataset\flickr8k\images"
handler = Words_Handler(min_frequency=5,
                        captions_path=captions_files_path)
custom_dataset = Data_set(parent_dir_path=parent_dir_path, captions_files_path=captions_files_path,
                          transform=image_transform)
data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, collate_fn=Customized_collate())

""" Import the model """
model = Image_captioner(embed_size=256, hidden_size=256,
                        captions_path=captions_files_path,
                        drop_prob=0.5)

""" Set Optimizer and Criterion """
learning_rate = 3e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss(ignore_index=handler.s2i["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if __name__ == "__main__":
    training = train(model, optimizer, criterion, data_loader, device, epochs)
