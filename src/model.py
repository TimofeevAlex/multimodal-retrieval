from torchvision import models
from transformers import DistilBertModel
from torch import nn
from torch import cuda

device = "cuda" if cuda.is_available() else "cpu"


class DistilBERT(nn.Module):
    def __init__(self, embedding_size, finetune=False):
        super(DistilBERT, self).__init__()
        self.distilBERT = DistilBertModel.from_pretrained("distilbert-base-uncased")

        for parameter in self.distilBERT.parameters():
            parameter.requires_grad = finetune

        self.last_layer = nn.Linear(768, embedding_size)

    def forward(self, input_ids, attention_mask):
        output_1 = self.distilBERT(input_ids=input_ids, attention_mask=attention_mask)[
            0
        ]
        pooler = output_1[:, 0]
        text_embeddings = self.last_layer(pooler)
        return text_embeddings


class ResNet34(nn.Module):
    def __init__(self, embedding_size, finetune=False):
        super(ResNet34, self).__init__()
        self.resnet = models.resnet34(pretrained=True)

        for parameter in self.resnet.parameters():
            parameter.requires_grad = finetune

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embedding_size)

    def forward(self, input_images):
        image_embeddings = self.resnet(input_images)
        return image_embeddings
