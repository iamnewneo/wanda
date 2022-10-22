import torch
import pickle
import joblib
import numpy as np
import torchvision.transforms as transforms

MEAN = [0.2112, 0.3528, 0.7894]
STD = [0.0742, 0.1805, 0.1066]


def switch_labels(a):
    a = np.array(a)
    return np.where((a == 0) | (a == 1), a ^ 1, a)


def save_object(obj, path):
    print(f"Saving: {obj.__class__.__name__} at: {path}")
    joblib.dump(obj, path)
    # with open(path, "wb") as outp:
    #     pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
    #     return True


def load_object(path):
    read_object = joblib.load(path)
    print(f"Loaded: {read_object.__class__.__name__} from: {path}")
    return read_object
    # read_object = None
    # with open(path, "rb") as inp:
    #     read_object = pickle.load(inp)
    #     print(f"Loaded: {read_object.__class__.__name__} from: {path}")
    # return read_object


def get_tranforms():
    return transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=MEAN, std=STD),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def un_norm_image(image_tensor):
    # unorm_tranform = UnNormalize(mean=MEAN, std=STD)
    # return unorm_tranform(image_tensor)
    return image_tensor * 255


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class NormalizeInverse(transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
