from pathlib import Path
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


class CorrosionSegDataset(Dataset):
    def __init__(self, root_dir, image_size=512, augment=False):
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / "images_512"
        self.masks_dir = self.root_dir / "mask_512"
        self.augment = augment

        image_paths = sorted(self.images_dir.glob("*.jpeg"))
        self.samples = self._build_samples(image_paths)

        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    def _build_samples(self, image_paths):
        samples = []
        missing_masks = []

        for image_path in image_paths:
            mask_path = self.masks_dir / f"{image_path.stem}.png"

            if mask_path.exists():
                samples.append((image_path, mask_path))
            else:
                missing_masks.append(mask_path.name)

        if missing_masks:
            preview = ", ".join(missing_masks[:5])
            raise FileNotFoundError(
                f"Missing {len(missing_masks)} masks in {self.masks_dir}: {preview}"
            )

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.augment:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if random.random() < 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
                
            if random.random() < 0.5:
                angle = random.uniform(-20, 20)
                image = TF.rotate(image, angle, interpolation=transforms.InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST)

            if random.random() < 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
                image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))            

        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0).float()

        return image, mask
