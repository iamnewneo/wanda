import torchvision.transforms as transforms


def get_tranforms():
    return transforms.Compose(
        [transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()]
    )
