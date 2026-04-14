import torch
import torchvision.models as models
import pandas as pd
from .preprocessing import preprocess_image

def load_model():
    """
    Load model from checkpoint with architecture: 25088→4096→1024→nb_classes
    """
    checkpoint = torch.load("models/best_vgg19_mushroom.pth", map_location="cpu")

    classes = checkpoint["classes"]
    nb_classes = len(classes)

    # Build VGG19 with custom classifier matching checkpoint
    model = models.vgg19(weights=None)
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(25088, 4096),   
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(4096, 1024),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(1024, nb_classes),
    )

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    df = pd.read_csv("Data/mushroom2.csv")

    desc_map = {
        row["scientific_name"]: {
            "description": row["new_description"],
            "edibility": row["edibility"]
        }
        for _, row in df.iterrows()
    }

    idx2info = {
        i: {
            "scientific_name": name,
            "description": desc_map.get(name, {}).get("description", ""),
            "edibility": desc_map.get(name, {}).get("edibility", "")
        }
        for i, name in enumerate(classes)
    }

    return model, idx2info


def predict(model, idx2info, image):
    tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).squeeze().numpy()

    idx = probs.argmax()

    return {
        "name": idx2info[idx]["scientific_name"],
        "description": idx2info[idx]["description"],
        "edibility": idx2info[idx]["edibility"],
        "confidence": float(probs[idx])
    }