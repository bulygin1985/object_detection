import torch
import torchvision.transforms.v2 as transforms
from tqdm import tqdm


def get_predictions(device, model, dataset, show_progress=False):
    """Get model predictions for the given dataset"""
    model.eval()
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )

    predictions = []
    if show_progress:
        pbar = tqdm(total=len(dataset))

    for img, _ in dataset:
        # Apply transformations
        img = transform(img)
        img = img.unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            pred = model(img)

        predictions.append(pred)
        pbar.update(1)

    return predictions
