import torch
from tqdm import tqdm


def generate_simple_data():
    train_features = torch.tensor([[1, 1], [1, 2], [2, 1], [8, 8], [8, 9], [9, 8], [5, 0], [5, 1], [6, 0], [6, 1]],
                                  dtype=torch.float32)
    train_labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=torch.long)

    test_features = torch.tensor([[1, 1.5], [8.5, 8.5], [5.5, 0.5]], dtype=torch.float32)
    test_labels = torch.tensor([0, 1, 2], dtype=torch.long)
    return train_features, train_labels, test_features, test_labels


def cosine_similarity(test_features, train_features):
    dot_product = torch.mm(test_features, train_features.t())
    test_norms = test_features.norm(dim=1, keepdim=True)
    train_norms = train_features.norm(dim=1, keepdim=True)
    cosine_sim = dot_product / (test_norms * train_norms.t())
    return cosine_sim


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=3):
    top1, top2, total = 0.0, 0.0, 0
    num_test_images, num_chunks = test_labels.shape[0], 5
    if num_chunks > num_test_images:
        num_chunks = num_test_images
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)

    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
                   idx: min((idx + imgs_per_chunk), num_test_images), :
                   ]
        targets = test_labels[idx: min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = cosine_similarity(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top2 = top2 + correct.narrow(1, 0, min(3, k)).sum().item()
        total += targets.size(0)

    top1 = top1 * 100.0 / total
    top2 = top2 * 100.0 / total
    return top1, top2


@torch.no_grad()
def extract_features(loader, model, device):
    features = []
    labels = []
    model.eval()
    for batch in tqdm(loader, desc="Extracting features"):
        images = batch['x'].to(device)
        batch_labels = batch['y'].to(device)
        batch_features = model(images)
        features.append(batch_features)
        labels.append(batch_labels)
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    if features.dim() > 2:
        features = features.squeeze()
    return features, labels


if __name__ == '__main__':
    # Simple test case
    train_features, train_labels, test_features, test_labels = generate_simple_data()
    k = 3
    T = 1.0
    knn_top1, knn_top2 = knn_classifier(train_features, train_labels, test_features, test_labels, k, T)
    print(f"KNN Top-1 Accuracy: {knn_top1:.2f}")
    print(f"KNN Top-2 Accuracy: {knn_top2:.2f}")
