import torch
import torchvision.transforms.v2 as transforms


def get_predictions(device, model, dataset):
    """Get model predictions for the given dataset"""
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )

    predictions = []
    for img, _ in dataset:
        # Apply transformations
        img = transform(img)
        img = img.unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            pred = model(img)

        predictions.append(pred)

    return predictions
