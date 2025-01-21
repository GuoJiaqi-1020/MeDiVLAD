import sys
import torch
import torch.nn.functional as F
from Medino.models.layer_dual_vlad import DualVLAD


class MediVLAD(torch.nn.Module):
    def __init__(self, embedding_backbone, input_size, hidden_size, seq_len, num_classes, num_clusters, **kwargs):
        super(MediVLAD, self).__init__()
        self.vec_len = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.attention = kwargs.get('attention', 'Gated')
        self.option = kwargs.get('option', False)
        self.warm_up = kwargs.get('warm_up', None)
        self.vlad_centroid = kwargs.get('vlad_centroid', None)

        # Load the pretrained model parameter
        self.Embedding = embedding_backbone

        for param in self.Embedding.parameters():
            param.requires_grad = True

        att_size = hidden_size

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(num_clusters*att_size, num_classes),
        )

        self.vlad = DualVLAD(
            num_clusters=num_clusters,
            dim=hidden_size,
            alpha=1.0,
            normalize_input=True,
            centroids=self.vlad_centroid)

    def forward(self, x, ep=0, dropout=0.0):
        batch_size = x.size(0)
        # Feature Extraction
        x = x.view(batch_size * self.seq_len, 3, 224, 224)

        if self.warm_up is not None and ep >= self.warm_up:
            img_emb = self.Embedding(x).view(batch_size, self.seq_len, self.vec_len)
        else:
            with torch.no_grad():
                img_emb = self.Embedding(x).view(batch_size, self.seq_len, self.vec_len)

        img_emb = torch.nn.functional.dropout(img_emb, p=dropout)
        x = self.vlad(img_emb)

        # Fully Connected Layer
        prob = self.dense(x)
        return prob


if __name__ == '__main__':
    kwargs = {}
    # model = vit_small(**kwargs)
    # calculate_model_complexity(model)
