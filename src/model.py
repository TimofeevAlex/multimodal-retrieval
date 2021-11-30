from torchvision import models
from transformers import DistilBertModel
from torch import nn
from torch import cuda

device = "cuda" if cuda.is_available() else "cpu"


class DistilBERT(nn.Module):
    def __init__(self, finetune=True, embedding_size=768):
        super(DistilBERT, self).__init__()
        self._finetune = finetune
        self.distilBERT = DistilBertModel.from_pretrained("distilbert-base-uncased")

        for parameter in self.distilBERT.parameters():
            parameter.requires_grad = (not finetune)

        if finetune:
            self.last_layer = nn.Linear(768, embedding_size)

    def forward(self, input_ids, attention_mask):
        output_1 = self.distilBERT(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )[0]
        text_embeddings = output_1[:, 0]
        if self._finetune:
            text_embeddings = self.last_layer(text_embeddings)
        return text_embeddings


class ResNet34(nn.Module):
    def __init__(self,  finetune=True, embedding_size=768):
        super(ResNet34, self).__init__()
        self._finetune = finetune
        self.resnet = models.resnet34(pretrained=True)

        for parameter in self.resnet.parameters():
            parameter.requires_grad = (not finetune)

        if finetune:
            last_layer = nn.Linear(self.resnet.fc.in_features, embedding_size)
        else:
            last_layer = nn.Sequential()

        self.resnet.fc = last_layer

    def forward(self, input_images):
        image_embeddings = self.resnet(input_images)
        return image_embeddings
