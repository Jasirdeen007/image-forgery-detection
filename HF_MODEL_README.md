---
library_name: pytorch
tags:
- image-classification
- computer-vision
- document-forgery-detection
- identity-document-analysis
license: mit
---

# EfficientNet-B3 Image Forgery Detection

This repository contains a fine-tuned EfficientNet-B3 model for binary classification of synthetic identity-document images as genuine or forged.

The checkpoint is provided as a PyTorch `state_dict`:

```text
efficientnet_b3_finetuned.pt
```

## Model description

- Architecture: EfficientNet-B3
- Framework: PyTorch and `timm`
- Task: binary image classification
- Input: one identity-document image
- Output: genuine or forged probability
- Input resolution: 300 × 300 pixels

Labels:

```text
0 = genuine / bona fide
1 = forged / tampered
```

## Dataset

The model was trained and evaluated on the template/composite portion of the Synthetic dataset of ID and Travel Documents (SIDTD), generated using MIDV-2020 document templates.

Dataset composition used in this project:

```text
Genuine images: 1,000
Forged images:  1,222
Total images:   2,222
```

The forged samples include the following SIDTD annotation types:

- `Crop_and_Replace`
- `Inpaint_and_Rewrite`

The dataset itself is not included in this repository. Please follow the SIDTD dataset terms and citation requirements before downloading or redistributing it.

## Preprocessing

Each image is:

1. converted to RGB;
2. resized to 300 × 300 pixels;
3. converted to a tensor;
4. normalized using ImageNet mean and standard deviation:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Training used mild brightness, contrast, and saturation variation. Horizontal flipping was not used because mirrored identity documents are not realistic examples.

## Training

The model was initialized from a pretrained EfficientNet-B3. Phase 1 trained the binary classifier head. The final checkpoint was then fine-tuned by unfreezing the final EfficientNet feature block and classifier head.

Fine-tuning used:

- optimizer: AdamW;
- learning rate: `1e-4`;
- weight decay: `1e-4`;
- epochs: 20;
- binary cross-entropy equivalent: two-class cross-entropy loss;
- deterministic stratified hold-out split: 80% train, 10% validation, 10% test.

## Evaluation

On the fixed 223-image test split at a decision threshold of `0.5`:

```text
Accuracy:  98.21%
Precision: 97.60%
Recall:    99.19%
F1:        98.39%
ROC-AUC:   99.98%
```

Confusion matrix, with rows representing actual labels and columns representing predictions:

```text
[[97,   3],
 [ 1, 122]]
```

Subtype recall on the forged test samples:

```text
Crop-and-replace: 94.74%
Inpainting:       100.00%
```

These results are specific to the project’s synthetic dataset and split. They should not be interpreted as real-world identity-verification performance.

## Usage

Install the required packages:

```bash
pip install torch torchvision timm pillow
```

Load the checkpoint:

```python
from pathlib import Path

import timm
import torch
from PIL import Image
from torchvision import transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = timm.create_model(
    "efficientnet_b3",
    pretrained=False,
    num_classes=2,
)
model.load_state_dict(
    torch.load(
        "efficientnet_b3_finetuned.pt",
        map_location="cpu",
        weights_only=True,
    )
)
model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])

image = Image.open("document.jpg").convert("RGB")
input_tensor = preprocess(image).unsqueeze(0).to(device)

with torch.no_grad():
    probabilities = torch.softmax(model(input_tensor), dim=1)[0]

genuine_probability = float(probabilities[0])
forged_probability = float(probabilities[1])
label = "forged" if forged_probability >= 0.5 else "genuine"

print({
    "label": label,
    "genuine_probability": genuine_probability,
    "forged_probability": forged_probability,
})
```

## Explainability

The project uses Grad-CAM with the final spatial convolutional layer of EfficientNet-B3. The resulting heatmap indicates image regions that contributed to the selected prediction.

Grad-CAM is an explanation aid and should not be interpreted as a guaranteed localization of the tampered region.

## Intended use

This model is intended for:

- educational computer-vision projects;
- research prototypes;
- synthetic identity-document forgery experiments;
- demonstrations of explainable image classification;
- offline error analysis.

## Limitations

- The training data is synthetic.
- The model has not been validated on real identity documents.
- Performance may depend on document templates, image quality, and dataset artifacts.
- The binary output is not equivalent to full identity verification.
- A positive or negative result should not be used as the sole basis for a KYC or access-control decision.
- The reported test split is an image-level random split and may not measure generalization to unseen document families as strictly as a grouped or external split.

## Citation and attribution

This model uses the SIDTD dataset and MIDV-2020-derived document templates. Please cite and follow the terms of the original SIDTD and MIDV-2020 resources when using this model or reproducing the experiments.

SIDTD project: https://github.com/Oriolrt/SIDTD_Dataset

## License

The model repository is released under the MIT license. Dataset licensing, attribution, and redistribution terms remain subject to the original SIDTD and MIDV-2020 sources.
