import torchvision.transforms as transforms


def get_tranforms():
    return transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
