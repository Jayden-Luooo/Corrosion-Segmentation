import random
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from torchvision import transforms
from src.models import UNet, ResNet34UNet
from datetime import datetime

MODEL_REGISTRY = {
    "unet": UNet,
    "resnet34_unet": ResNet34UNet,
}


def build_model(model_name, in_channels=3, out_channels=1, base_channels=64, bilinear=False):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_name: {model_name}. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )

    model_class = MODEL_REGISTRY[model_name]
    return model_class(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        bilinear=bilinear,
    )


def load_model(
    checkpoint_path,
    device,
    model_name="unet",
    in_channels=3,
    out_channels=1,
    base_channels=64,
    bilinear=False,
):
    model = build_model(
        model_name=model_name,
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        bilinear=bilinear,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def predict_mask(model, image_path, device, image_size=512, threshold=0.58):
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        prob = torch.sigmoid(logits)
        mask = (prob > threshold).float()

    mask = transforms.ToPILImage()(mask.squeeze().cpu())
    return image.resize((image_size, image_size)), mask

def make_overlay(image, mask, alpha=120):
    overlay = image.copy().convert("RGBA")

    red_mask = Image.new("RGBA", image.size, (255, 0, 0, 0))
    red_mask.putalpha(mask.point(lambda p: alpha if p > 0 else 0))

    overlay = Image.alpha_composite(overlay, red_mask)
    return overlay


def plot_random_predictions(
    model,
    dataset,
    device,
    image_size=512,
    threshold=0.58,
    num_samples=3,
    seed=None,
):
    if seed is not None:
        random.seed(seed)

    num_samples = min(num_samples, len(dataset))
    sample_indices = random.sample(range(len(dataset)), num_samples)

    plt.figure(figsize=(18, 5 * num_samples))

    for row, idx in enumerate(sample_indices):
        image_path, _ = dataset.samples[idx]

        raw_image, pred_mask = predict_mask(
            model=model,
            image_path=image_path,
            device=device,
            image_size=image_size,
            threshold=threshold,
        )

        true_mask = dataset[idx][1].squeeze()
        overlay = make_overlay(raw_image, pred_mask)

        plt.subplot(num_samples, 4, row * 4 + 1)
        plt.imshow(raw_image)
        plt.title("Raw Image")
        plt.axis("off")

        plt.subplot(num_samples, 4, row * 4 + 2)
        plt.imshow(true_mask, cmap="gray")
        plt.title("True Mask")
        plt.axis("off")

        plt.subplot(num_samples, 4, row * 4 + 3)
        plt.imshow(pred_mask, cmap="gray")
        plt.title("Predicted Mask")
        plt.axis("off")

        plt.subplot(num_samples, 4, row * 4 + 4)
        plt.imshow(overlay)
        plt.title("Prediction Overlay")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def save_random_prediction_figures(
    model,
    dataset,
    device,
    save_dir="outputs/predicts",
    image_size=512,
    threshold=0.58,
    num_samples=5,
    seed=42,
    checkpoint_path=None,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    sample_indices = rng.sample(range(len(dataset)), min(num_samples, len(dataset)))

    log_lines = [
        f"time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"checkpoint: {checkpoint_path}",
        f"dataset: {dataset.root_dir}",
        f"image_size: {image_size}",
        f"threshold: {threshold}",
        f"num_saved: {len(sample_indices)}",
        "",
    ]

    for order, index in enumerate(sample_indices, start=1):
        image_path, mask_path = dataset.samples[index]

        raw_image, pred_mask = predict_mask(
            model=model,
            image_path=image_path,
            device=device,
            image_size=image_size,
            threshold=threshold,
        )

        true_mask = Image.open(mask_path).convert("L")
        true_mask = true_mask.resize((image_size, image_size), resample=Image.NEAREST)
        overlay = make_overlay(raw_image, pred_mask)

        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        items = [
            ("Raw Image", raw_image, {}),
            ("True Mask", true_mask, {"cmap": "gray"}),
            ("Predicted Mask", pred_mask, {"cmap": "gray"}),
            ("Prediction Overlay", overlay, {}),
        ]

        for ax, (title, item, kwargs) in zip(axes, items):
            ax.imshow(item, **kwargs)
            ax.set_title(title)
            ax.axis("off")

        save_file = save_dir / f"predict_{order:02d}_{image_path.stem}.png"

        plt.tight_layout()
        fig.savefig(save_file, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        log_lines.append(
            f"{order:02d} index={index} image={image_path.name} "
            f"mask={mask_path.name} output={save_file.name}"
        )

    log_path = save_dir / "predict_log.txt"
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    return save_dir, log_path
