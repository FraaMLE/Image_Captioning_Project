import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10,kernel_size=1,stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=20, out_channels=30, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=30, out_channels=40, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=40, out_channels=50, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=50, out_channels=60, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, padding=0, stride=2),
            nn.Sigmoid()
            )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=60, out_channels=40, stride=2, kernel_size=5),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=40, out_channels=20, stride=2, kernel_size=5),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=20, out_channels=10, stride=2, kernel_size=5),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=10, out_channels=3, stride=1, kernel_size=8)
        )

        self.apply(self.weights_initialization)

    def weights_initialization(self, module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            nn.init.normal_(module.weight.data, mean=0.0, std=0.1)
            if module.bias is not None:
                nn.init.zeros_(module.bias.data)

    def manifold(self, x):
        with torch.no_grad():
            return self.encoder(x)


    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == "__main__":
    batch_size = 3; sequence_length = 12
    channels = 3
    height = 300
    width = 300
    emb = 256
    batch = torch.empty(batch_size, channels, height, width)
    shape = (3, 300, 300)
    single_image = torch.rand(*shape)
    encoder = ConvAutoencoder()
    enc_im = encoder.manifold(single_image)
    print(enc_im.reshape(-1).shape)
    # encoded_batch = encoder.manifold(batch)
    # reshaped_batch = encoded_batch.view(batch_size, -1)
    mapper = nn.Sequential(nn.Linear(34*34*60, emb*5))
