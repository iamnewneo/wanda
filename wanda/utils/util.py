import torchvision.transforms as transforms


def get_tranforms():
    return transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(256)])
