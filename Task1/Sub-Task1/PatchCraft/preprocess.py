import torch
import torchvision.transforms.v2
import numpy as np
import random
from torchvision.utils import make_grid


def create_patches(img, patch_size=32):
    B, C, H, W = img.shape
    patches = img.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
    patches = patches.permute(0, 2, 4, 1, 3, 5)
    patches = patches.reshape(B, -1, C, patch_size, patch_size)
    return patches


def layer1(patch):
    cum_sum = (patch[:, :, :, :-1] - patch[:, :, :, 1:]).sum(dim = (1,2,3))
    return cum_sum


def layer2(patch):
    cum_sum = (patch[:, :, -1:, :] - patch[:, :, 1:, :]).sum(dim = (1, 2, 3))
    return cum_sum



def layer3and4(patch):
    cum_sum_l3 = (patch[:, :, :-1, :-1]  - patch[:, :, 1:, 1:]).sum(dim = (1, 2, 3))
    cum_sum_l4 = (patch[:, :, :-1, 1:] - patch[:, :, 1:, :-1]).sum(dim = (1, 2, 3))
    return cum_sum_l3 + cum_sum_l4


def cal_var(patch):
    final_var = layer1(patch) + layer2(patch) + layer3and4(patch)
    return final_var


def rich_and_poor(patch_list):
    # Compute variations for all patches across the batch
    # Layer 1: Compute cumulative sum along the last width axis
    layer1_sum = (patch_list[:, :, :, :, :-1] - patch_list[:, :, :, :, 1:]).sum(dim=(2, 3, 4))
    layer2_sum = (patch_list[:, :, :, -1:, :] - patch_list[:, :, :, 1:, :]).sum(dim=(2, 3, 4))
    layer3_sum = (patch_list[:, :, :, :-1, :-1] - patch_list[:, :, :, 1:, 1:]).sum(dim=(2, 3, 4))
    layer4_sum = (patch_list[:, :, :, -1:, 1:] - patch_list[:, :, :, 1:, :-1]).sum(dim=(2, 3, 4))
    variations = layer1_sum + layer2_sum + layer3_sum + layer4_sum

    thresholds = variations.mean(dim=1, keepdim=True)
    rich_list = []
    poor_list =[]
    for b_id in range(patch_list.size()[0]):
        patches = patch_list[b_id]
        vars = cal_var(patches)
        rich_mask = vars >= thresholds[b_id]
        rich_patches = patches[rich_mask]
        rich_list.append(rich_patches)
        poor_mask = vars < thresholds[b_id]
        poor_patches = patches[poor_mask]
        poor_list.append(poor_patches)

    return rich_list, poor_list



def make_grid_patches(patch_list):
    grids = []

    for tensor in patch_list:
        patch_list = list(tensor)  # Convert to a list for manipulation

        # Ensure at least 64 patches by duplicating random patches if necessary
        while len(patch_list) < 64:
            random_patch = patch_list[random.randrange(0, len(patch_list))]
            patch_list.append(random_patch)

        # Create a grid from the patches
        grid_tensor = make_grid(patch_list, nrow=8, padding=0)  # (C, Grid_Height, Grid_Width)
        grids.append(grid_tensor[0].unsqueeze(0))  # Keep only one channel (assumes grayscale)

    # Stack all batch grids
    return torch.stack(grids)


def preprocess_image(img):
    patch_list = create_patches(img)
    rich_list, poor_list = rich_and_poor(patch_list)
    rich_grid = make_grid_patches(rich_list)
    poor_grid = make_grid_patches(poor_list)
    return rich_grid, poor_grid


if __name__ == "__main__":
    img = torch.randn(1, 1, 256, 256)
    img,_ = preprocess_image(img)
    print(img.size())
