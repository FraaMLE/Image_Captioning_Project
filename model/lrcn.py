import torch
import torch.nn as nn
import torchvision.models
import torchvision.models as models
import torchvision.transforms as transforms
from data_handler.words_handler import Words_Handler
from data_handler.data_loading import ConcatenateFeaturesAndSequences, Data_set, Customized_collate
from torch.utils.data import DataLoader, Dataset


class Lrcn(nn.Module):
    def __init__(self, embed_size: int, path_captions: str, min_frequency: int):
        super(Lrcn, self).__init__()
        self.vocab = Words_Handler(min_frequency=min_frequency,
                                   captions_path=path_captions)
        # ENCODER: CNN
        # N x 3 x 299 x 299, so ensure your images are sized accordingly.
        self.inception = models.inception_v3(init_weights=True)


        self.relu = nn.ReLU()
        self.dropout_01 = nn.Dropout(0.5)

        # DECODER: LSTM
        # vocab_size --> rows ||| embded_size --> columns
        self.embed = nn.Embedding(len(self.vocab.s2i), embed_size)  # Not pre-trained
        self.concatenator = ConcatenateFeaturesAndSequences(embed_size)
        # LSTM_input_size, LSTM_hidden_state_size_01 = embed_size
        # embed_size_img + embded_size_word = embed_size * 2
        self.lstm_01 = nn.LSTM(embed_size * 2, embed_size, num_layers=1, batch_first=True)
        self.lstm_02 = nn.LSTM(embed_size * 2, embed_size, num_layers=1, batch_first=True)
        self.linear = nn.Linear(embed_size, len(self.vocab.s2i))
        self.dropout_02 = nn.Dropout(0.5)

    def forward(self, images, captions):
        """
        ENCODER:
        A batch of images is passed through the CNN that extract the features of the images
        ---> CNN_output: (N, features_size), where features_size == embed_size
        Then each features is replicated in order to match the length of the sequence so that the batch
        of caption can be concatenated in with the images.
        ---> replicated_CNN_output: (N, L, features_size)
        The corresponding batch of captures of shape (N, L, 1) is then mapped into the embedding space
        becoming (N, L, embed_size).
        Finally, the embeddings and the features are concatenated and passed to the decoder.

        DECODER:
        The second LSTM takes as input the output of the first one (whose size coincides with the embed_size)
        concatenated with replicated_CNN_output, and it initializes
        its h_0 and c_0 with the one output by the last recurrence of the LSTM_1.
        """
        features, _ = self.inception(images)
        features = self.dropout_01(self.relu(features))

        # (N, sequence_length, embed_size)
        embedded_captions = self.dropout_02(
            self.embed(captions)
        )
        # add concatenation here: (N, sequence_length, embed_size*2)
        concatenated_input = self.concatenator(features, embedded_captions)
        # lstm_1: (N, L, LSTM_hidden_state_size_01) where LSTM_hidden_state_size_01 == embed_size
        lstm_1, (h_0_1, c_0_1) = self.lstm_01(concatenated_input)
        # lstm_1 concatenated with features: (N, sequence_length, embed_size*2)
        lstm_1_concatenated = self.concatenator(features, lstm_1)
        lstm_2, _ = self.lstm_02(lstm_1_concatenated, (h_0_1, c_0_1))

        outputs = self.linear(lstm_2)
        return outputs

    def generate_text_capture(self, image, max_length=40):
        text_caption = []
        sos_tensor = torch.full((image.shape[0], 1), self.vocab.s2i['<SOS>'])

        with torch.no_grad():
            features, _ = self.inception(image)
            features = self.dropout_01(self.relu(features))
            states_01 = None

            for _ in range(max_length):
                if _ == 0:
                    start_text = self.dropout_02(
                        self.embed(sos_tensor)
                    )
                    concatenation_01 = torch.cat((features, start_text), dim=0)
                    lstm_1, (h_0_1, c_0_1) = self.lstm_01(concatenation_01)
                    concatenation_02 = torch.cat((features, lstm_1), dim=0)
                    lstm_2, _ = self.lstm_02(concatenation_02, (h_0_1, c_0_1))
                    outputs = self.linear(lstm_2)
                    prediction = outputs.argmax(1)
                    text_caption.append(prediction.item())
                    text = self.dropout_02(
                        self.embed(start_text)
                    )
                    if self.vocab.i2s[prediction.item()] == "<EOS>":
                        break
                else:
                    concatenation_01 = torch.cat((features, text), dim=0)
                    lstm_1, (h_0_1, c_0_1) = self.lstm_01(concatenation_01)
                    concatenation_02 = torch.cat((features, lstm_1), dim=0)
                    lstm_2, _ = self.lstm_02(concatenation_02, (h_0_1, c_0_1))
                    outputs = self.linear(lstm_2)
                    prediction = outputs.argmax(1)
                    text_caption.append(prediction.item())
                    text = embedded_captions = self.dropout_02(
                        self.embed(text)
                    )
                    if self.vocab.i2s[prediction.item()] == "<EOS>":
                        break
        return [self.vocab.i2s[idx] for idx in text_caption]




if __name__ == "__main__":
    captions_files_path = r"C:\Users\franv\Downloads\Images_Dataset\flickr8k\captions.txt"
    parent_dir_path = r"C:\Users\franv\Downloads\Images_Dataset\flickr8k\images"
    # Step 1: Define image transforms (adjust to your needs)
    image_transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    custom_dataset = Data_set(parent_dir_path
                              =parent_dir_path,
                              captions_files_path=captions_files_path,
                              transform=image_transform)
    batch_size = 32  # Adjust this according to your needs
    data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, collate_fn=Customized_collate())
    model = Lrcn(embed_size=256, path_captions=captions_files_path,
                 min_frequency=5)

    for images, captions in data_loader:
        t = model.generate_text_capture(images)
        print(t)
        print(captions.shape)
        print(type(captions))
        break