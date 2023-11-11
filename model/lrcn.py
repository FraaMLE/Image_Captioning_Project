import torch
import torch.nn as nn
import torchvision.models
import torchvision.models as models
import torchvision.transforms as transforms
from data_handler.words_handler import Words_Handler
from data_handler.data_loading import ConcatenateFeaturesAndSequences, Data_set, Customized_collate
from torch.utils.data import DataLoader, Dataset
from convolutional_autoencoder import ConvAutoencoder


class Encoder(nn.Module):
    def __init__(self, embed_size: int, drop_prob: float, manifold_size=34 * 34 * 60):
        super(Encoder, self).__init__()
        self.conv_autoenc = ConvAutoencoder().eval()
        self.linear_mapper = nn.Sequential(nn.Linear(manifold_size, embed_size * 5),
                                           nn.Linear(embed_size * 5, embed_size * 3),
                                           nn.Linear(embed_size * 3, embed_size))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, images):
        manifold = self.conv_autoenc.manifold(images)
        reshaped_manifold = manifold.view(images.size(0), -1)
        features = self.linear_mapper(reshaped_manifold)
        return self.dropout(self.relu(features))


class Decoder(nn.Module):
    def __init__(self, embed_size: int, h: int, vocab_size: int, drop_prob: float):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm_01 = nn.LSTM(embed_size * 2, embed_size, num_layers=1, batch_first=True)
        self.lstm_02 = nn.LSTM(embed_size * 2, embed_size, num_layers=1, batch_first=True)
        self.linear = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(drop_prob)
        self.concatenate = ConcatenateFeaturesAndSequences(embed_size)
        self.embed_size = embed_size

    def forward(self, features, caption):
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

    def texter(self, x, word, flag=True):
        if flag:
            word = torch.IntTensor([word])
        with torch.no_grad():
            word = self.decoder.dropout(self.decoder.embed(word))
            con_01 = torch.cat((x, word), dim=1)
            lstm_1, (h_0_1, c_0_1) = self.decoder.lstm_01(con_01)
            lstm_1_concatenated = torch.cat((x, lstm_1), dim=1)
            lstm_2, _ = self.decoder.lstm_02(lstm_1_concatenated, (h_0_1, c_0_1))
            outputs = self.decoder.linear(lstm_2)
            return outputs

    def generate_text_capture(self, image, max_length=40):
        text_caption = []
        with torch.no_grad():
            x = self.encoder(image)
            for iteration in range(max_length):
                if iteration == 0:
                    sos = self.vocab.s2i["<SOS>"]
                    outputs = self.texter(x, sos)  # [1, 2664] --> [2664], correct?
                    prediction = outputs.argmax(1)
                    text_caption.append(prediction.item())

                else:
                    if text_caption[-1] != self.vocab.s2i["<EOS>"]:
                        outputs = self.texter(x, text_caption[-1])
                        prediction = outputs.argmax(1)
                        text_caption.append(prediction.item())
                    else:
                        return text_caption
        return text_caption


if __name__ == "__main__":
    batch_size = 3
    sequence_length = 12
    channels = 3
    height = 300
    width = 300
    emb = 256
    batch = torch.empty(batch_size, channels, height, width)
    cap = torch.LongTensor(batch_size, sequence_length).random_(0, 100)
    mo = Image_captioner(embed_size=256, hidden_size=256,
                         captions_path=r"C:\Users\franv\Downloads\Images_Dataset\flickr8k\captions.txt",
                         drop_prob=0.5)
    print(mo(batch, cap).shape)  # torch.Size([batch_size, sequence_length, vocabulary size])

    # word = torch.LongTensor(1).random_(0,2260)
    # shape = (3, 300, 300)
    # single_image = torch.rand(*shape)
    # em = Encoder(embed_size=256, drop_prob=0.5).eval()
    # ima = em(single_image.unsqueeze(0))
    # print(ima.shape)
    # x = torch.randn(256)
    # pr = mo.texter(ima, word)
    # print(pr.shape)
    # print(word.shape)
    # shape = (1, 3, 300, 300)
    # single_image = torch.rand(*shape)
    # proc = mo.encoder(single_image)
    # print(proc.shape)
    # idx = mo.vocab.s2i["<SOS>"]
    # mapped = mo.decoder.embed(torch.LongTensor([idx]))
    # print(mo.texter(proc, idx).argmax(1)) # tensor([103])
    # t = mo.generate_text_capture(single_image)
    # print(t)

    # print(mo.texter(x, word))
    # print(x.shape)
