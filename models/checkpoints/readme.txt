It is a placeholder for pretrained model weights.

You can load save model weights with the following python code (for cuda execution):

model.load_state_dict(torch.load("pretrained_weights_overfit.pt", weights_only=True, map_location="cuda:0"))
