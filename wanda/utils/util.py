import torchvision.transforms as transforms


def get_tranforms():
    return transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.2112, 0.3528, 0.7894], std=[0.0742, 0.1805, 0.1066]
            ),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
