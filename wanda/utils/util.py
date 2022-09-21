import torchvision.transforms as transforms


def get_tranforms():
    return transforms.Compose([transforms.CenterCrop(256), transforms.ToTensor()])
