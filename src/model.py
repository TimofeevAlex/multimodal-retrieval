from torchvision import models
from transformers import DistilBertModel
from torch import nn
from torch import cuda

device = "cuda" if cuda.is_available() else "cpu"


class DistilBERT(nn.Module):
    def __init__(self, finetune=None, embedding_size=768):
        super(DistilBERT, self).__init__()
        self._finetune = finetune
        self.distilBERT = DistilBertModel.from_pretrained("distilbert-base-uncased")

        for parameter in self.distilBERT.parameters():
            parameter.requires_grad = (finetune == "all")

        if finetune != None:
            self.last_layer = nn.Linear(768, embedding_size)

    def forward(self, input_ids, attention_mask):
        output_1 = self.distilBERT(input_ids=input_ids, attention_mask=attention_mask)[
            0
        ]
        text_embeddings = output_1[:, 0]
        if self._finetune != None:
            text_embeddings = self.last_layer(text_embeddings)
        return text_embeddings


class ResNet50(nn.Module):
    def __init__(self, finetune=None, embedding_size=768):
        super(ResNet50, self).__init__()
        self._finetune = finetune
        self.resnet = models.resnet50(pretrained=True)

        for parameter in self.resnet.parameters():
            parameter.requires_grad = (finetune == "all")

        if finetune == None:
            last_layer = nn.Sequential()
        else:
            last_layer = nn.Linear(self.resnet.fc.in_features, embedding_size)

        self.resnet.fc = last_layer

    def forward(self, input_images):
        image_embeddings = self.resnet(input_images)
        return image_embeddings
