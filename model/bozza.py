import torch
import torch.nn as nn
import torchvision.models
import torchvision.models as models
import torchvision.transforms as transforms
from data_handler.words_handler import Words_Handler
from data_handler.data_loading import ConcatenateFeaturesAndSequences, Data_set, Customized_collate
from torch.utils.data import DataLoader, Dataset


class Encoder(nn.Module):
    def __init__(self, embed_size: int, drop_prob: float):
        super(Encoder, self).__init__()
        self.inception = models.inception_v3(init_weights=True)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, images):
        features, _ = self.inception(images)
        return self.dropout(self.relu(features))



class Decoder(nn.Module):
    def __init__(self, embed_size: int, h: int, vocab_size: int, drop_prob: float):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm_01 = nn.LSTM(embed_size*2, embed_size, num_layers=1, batch_first=True)
        self.lstm_02 = nn.LSTM(embed_size*2, embed_size, num_layers=1, batch_first=True)
        self.linear = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(drop_prob)
        self.concatenate = ConcatenateFeaturesAndSequences(embed_size)
        self.embed_size = embed_size

    def forward(self, features,  caption):
        mapped_words = self.dropout(self.embed(caption))
        con_01 = self.concatenate(features, mapped_words)
        # lstm_1: (N, L, LSTM_hidden_state_size_01) where LSTM_hidden_state_size_01 == embed_size
        lstm_1, (h_0_1, c_0_1) = self.lstm_01(con_01)
        # lstm_1 concatenated with features: (N, sequence_length, embed_size*2)
        lstm_1_concatenated = self.concatenate(features, lstm_1)
        lstm_2, _ = self.lstm_02(lstm_1_concatenated, (h_0_1, c_0_1))
        outputs = self.linear(lstm_2)
        return outputs




class Image_captioner(nn.Module):
    def __init__(self, embed_size, hidden_size, captions_path, drop_prob=0.5):
        super(Image_captioner, self).__init__()
        self.vocab = Words_Handler(min_frequency=5, captions_path=captions_path)
        self.encoder = Encoder(embed_size, drop_prob)
        self.decoder = Decoder(embed_size, hidden_size, vocab_size=len(self.vocab.s2i), drop_prob=drop_prob)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs


    def generate_text_capture(self, image, max_length=40):
        pass




batch_size = 3; sequence_length = 12
channels = 3
height = 299
width = 299
emb = 256
batch = torch.empty(batch_size, channels, height, width)
mo = Image_captioner(embed_size=256, hidden_size=256,
                     captions_path=r"C:\Users\franv\Downloads\Images_Dataset\flickr8k\captions.txt",
                     drop_prob=0.5)
cap = torch.LongTensor(batch_size, sequence_length).random_(0,100)
mo(batch, cap)

inc = models.inception_v3(init_weights=True)



# captions_files_path = r"C:\Users\franv\Downloads\Images_Dataset\flickr8k\captions.txt"
# parent_dir_path = r"C:\Users\franv\Downloads\Images_Dataset\flickr8k\images"
# # Step 1: Define image transforms (adjust to your needs)
# image_transform = transforms.Compose(
#     [
#         transforms.Resize((356, 356)),
#         transforms.RandomCrop((299, 299)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ]
# )
# custom_dataset = Data_set(parent_dir_path
#                           =parent_dir_path,
#                           captions_files_path=captions_files_path,
#                           transform=image_transform)
# batch_size = 32  # Adjust this according to your needs
# data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, collate_fn=Customized_collate())
#
#
# for images, captions in data_loader:
#     cnn.eval()
#     print(cnn(images[0].unsqueeze(0)))
#     break
