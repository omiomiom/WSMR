from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets.utils import download_url

from .utils import seed_worker


IMAGENET_DEVKIT = "ILSVRC2012_devkit_t12.tar.gz"
IMAGENET_DEVKIT_URL = (
    "https://image-net.org/data/ILSVRC/2012/"
    "ILSVRC2012_devkit_t12.tar.gz"
)
IMAGENET_DEVKIT_MD5 = "fa75699e90414af021442c21a62c3abf"
IMAGENET_TRAIN_ARCHIVE = "ILSVRC2012_img_train.tar"
IMAGENET_VAL_ARCHIVE = "ILSVRC2012_img_val.tar"


@dataclass
class DataBundle:
    train_loader: DataLoader
    calibration_loader: DataLoader
    eval_loader: DataLoader
    train_generator: torch.Generator
    num_classes: int
    input_shape: tuple[int, int, int]
    mean: tuple[float, ...]
    std: tuple[float, ...]


def _mnist(config: dict):
    root = Path(config["paths"]["dataset_root"])
    transform = transforms.ToTensor()
    train = datasets.MNIST(
        root=root,
        train=True,
        transform=transform,
        download=bool(config["dataset_options"]["download"]),
    )
    test = datasets.MNIST(
        root=root,
        train=False,
        transform=transform,
        download=bool(config["dataset_options"]["download"]),
    )
    return train, test, 10, (1, 28, 28), (0.1307,), (0.3081,)


def _cifar100(config: dict):
    root = Path(config["paths"]["dataset_root"])
    if bool(config["dataset_options"]["train_augmentation"]):
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    else:
        train_transform = transforms.ToTensor()
    eval_transform = transforms.ToTensor()
    train = datasets.CIFAR100(
        root=root,
        train=True,
        transform=train_transform,
        download=bool(config["dataset_options"]["download"]),
    )
    test = datasets.CIFAR100(
        root=root,
        train=False,
        transform=eval_transform,
        download=bool(config["dataset_options"]["download"]),
    )
    return (
        train,
        test,
        100,
        (3, 32, 32),
        (0.5071, 0.4867, 0.4408),
        (0.2675, 0.2565, 0.2761),
    )


def _prepare_imagenet1k(root: Path, options: dict) -> None:
    """Turn the official ILSVRC2012 archives into ImageFolder splits.

    Torchvision's ImageNet parser verifies the official archive checksums,
    reads the validation ground truth from the devkit, and puts validation
    images into synset directories.  In particular, validation labels must
    never be inferred from the order of members in the image archive.
    """

    train_root = root / "train"
    val_root = root / "val"
    if train_root.is_dir() and val_root.is_dir():
        return

    missing_archives = [
        filename
        for filename in (IMAGENET_TRAIN_ARCHIVE, IMAGENET_VAL_ARCHIVE)
        if not (root / filename).is_file()
    ]
    if missing_archives:
        raise FileNotFoundError(
            f"ImageNet-1k is not prepared under {root}. Expected either "
            "train/ and val/ class folders, or the official archives "
            f"{IMAGENET_TRAIN_ARCHIVE} and {IMAGENET_VAL_ARCHIVE}. "
            f"Missing: {', '.join(missing_archives)}."
        )

    if not bool(options.get("prepare_archives", True)):
        raise FileNotFoundError(
            f"ImageNet archives were found under {root}, but automatic "
            "preparation is disabled by dataset_options.prepare_archives."
        )

    devkit = root / IMAGENET_DEVKIT
    if not devkit.is_file():
        if not bool(options.get("download_devkit", False)):
            raise FileNotFoundError(
                f"{devkit} is required to assign the 50,000 validation "
                "images to their correct classes. Add the official devkit "
                "there or enable dataset_options.download_devkit."
            )
        root.mkdir(parents=True, exist_ok=True)
        print(f"downloading ImageNet validation metadata to {devkit}")
        try:
            download_url(
                IMAGENET_DEVKIT_URL,
                str(root),
                filename=IMAGENET_DEVKIT,
                md5=IMAGENET_DEVKIT_MD5,
            )
        except Exception as error:
            raise RuntimeError(
                "Could not download the official ImageNet devkit. "
                f"Download {IMAGENET_DEVKIT} from the ImageNet ILSVRC2012 "
                f"download page and place it under {root}."
            ) from error

    print(
        "preparing ImageNet-1k archives (first run only; this needs about "
        "145 GB of additional disk space)"
    )
    # Prepare checksum-verified archives before creating ImageFolder datasets
    # with the requested transforms.
    datasets.ImageNet(root=str(root), split="train")
    datasets.ImageNet(root=str(root), split="val")


def _validate_imagenet1k(train, val, root: Path) -> None:
    if len(train.classes) != 1000:
        raise RuntimeError(
            f"Expected 1000 ImageNet training classes under {root / 'train'}, "
            f"but found {len(train.classes)}."
        )
    if train.class_to_idx != val.class_to_idx:
        train_names = set(train.class_to_idx)
        val_names = set(val.class_to_idx)
        missing = sorted(train_names - val_names)
        extra = sorted(val_names - train_names)
        details = []
        if missing:
            details.append(f"missing from val: {', '.join(missing[:5])}")
        if extra:
            details.append(f"only in val: {', '.join(extra[:5])}")
        raise RuntimeError(
            "ImageNet train/val class mappings do not match under "
            f"{root} ({'; '.join(details)})."
        )


def _imagenet1k(config: dict):
    root = Path(config["paths"]["dataset_root"])
    options = config["dataset_options"]
    _prepare_imagenet1k(root, options)
    train_root = root / "train"
    val_root = root / "val"
    if bool(options["train_augmentation"]):
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    train = datasets.ImageFolder(train_root, transform=train_transform)
    val = datasets.ImageFolder(val_root, transform=eval_transform)
    _validate_imagenet1k(train, val, root)
    return (
        train,
        val,
        1000,
        (3, 224, 224),
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    )


def build_dataloaders(config: dict) -> DataBundle:
    dataset_name = config["dataset"]
    if dataset_name == "mnist":
        result = _mnist(config)
    elif dataset_name == "cifar100":
        result = _cifar100(config)
    elif dataset_name == "imagenet1k":
        result = _imagenet1k(config)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_set, eval_set, classes, shape, mean, std = result
    seed = int(config["experiment"]["seed"])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    calibration_generator = torch.Generator(device="cpu")
    calibration_generator.manual_seed(seed + 1)
    evaluation_generator = torch.Generator(device="cpu")
    evaluation_generator.manual_seed(seed + 2)
    workers = int(config["loader"]["num_workers"])
    persistent = (
        bool(config["loader"]["persistent_workers"]) and workers > 0
    )
    common = {
        "num_workers": workers,
        "pin_memory": bool(config["loader"]["pin_memory"]),
        "persistent_workers": persistent,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_set,
        batch_size=int(config["loader"]["batch_size"]),
        shuffle=True,
        drop_last=False,
        generator=generator,
        **common,
    )
    # A separate non-shuffled loader prevents BN calibration from advancing
    # the training sampler generator and changing the next epoch's data order.
    calibration_loader = DataLoader(
        train_set,
        batch_size=int(config["loader"]["eval_batch_size"]),
        shuffle=False,
        drop_last=False,
        generator=calibration_generator,
        **common,
    )
    eval_loader = DataLoader(
        eval_set,
        batch_size=int(config["loader"]["eval_batch_size"]),
        shuffle=False,
        drop_last=False,
        generator=evaluation_generator,
        **common,
    )
    return DataBundle(
        train_loader=train_loader,
        calibration_loader=calibration_loader,
        eval_loader=eval_loader,
        train_generator=generator,
        num_classes=classes,
        input_shape=shape,
        mean=mean,
        std=std,
    )
