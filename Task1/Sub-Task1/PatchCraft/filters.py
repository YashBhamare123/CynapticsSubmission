import torch
import torch.nn.functional as F

filter_bank_a = torch.tensor(
                    [[[ 0,  0,  0,  0,  0],
                    [ 0,  1,  0,  0,  0],
                    [ 0,  0, -1,  0,  0],
                    [ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  0,  0]],

                   [[ 0,  0,  0,  0,  0],
                    [ 0,  0,  1,  0,  0],
                    [ 0,  0, -1,  0,  0],
                    [ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  0,  0]],

                   [[ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  1,  0],
                    [ 0,  0, -1,  0,  0],
                    [ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  0,  0]],

                   [[ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  0,  0],
                    [ 0,  0, -1,  1,  0],
                    [ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  0,  0]],

                   [[ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  0,  0],
                    [ 0,  0, -1,  0,  0],
                    [ 0,  0,  0,  1,  0],
                    [ 0,  0,  0,  0,  0]],

                   [[ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  0,  0],
                    [ 0,  0, -1,  0,  0],
                    [ 0,  0,  1,  0,  0],
                    [ 0,  0,  0,  0,  0]],

                   [[ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  0,  0],
                    [ 0,  0, -1,  0,  0],
                    [ 0,  1,  0,  0,  0],
                    [ 0,  0,  0,  0,  0]],

                   [[ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  0,  0],
                    [ 0,  1, -1,  0,  0],
                    [ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  0,  0]]], dtype=torch.float32)/float(255)

filter_bank_b = torch.tensor(
                    [[[ 0,  0,  0,  0,  0],
                    [ 0,  2,  1,  0,  0],
                    [ 0,  1, -3,  0,  0],
                    [ 0,  0,  0,  1,  0],
                    [ 0,  0,  0,  0,  0]],

                    [[ 0,  0, -1,  0,  0],
                    [ 0,  0,  3,  0,  0],
                    [ 0,  0, -3,  0,  0],
                    [ 0,  0,  1,  0,  0],
                    [ 0,  0,  0,  0,  0]],

                    [[ 0,  0,  0,  0,  0],
                    [ 0,  0,  1,  2,  0],
                    [ 0,  0, -3,  1,  0],
                    [ 0,  1,  0,  0,  0],
                    [ 0,  0,  0,  0,  0]],

                    [[ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  0,  0],
                    [ 0,  1, -3,  3, -1],
                    [ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  0,  0]],

                    [[ 0,  0,  0,  0,  0],
                    [ 0,  1,  0,  0,  0],
                    [ 0,  0, -3,  1,  0],
                    [ 0,  0,  1,  2,  0],
                    [ 0,  0,  0,  0,  0]],

                    [[ 0,  0,  0,  0,  0],
                    [ 0,  0,  1,  0,  0],
                    [ 0,  0, -3,  0,  0],
                    [ 0,  0,  3,  0,  0],
                    [ 0,  0, -1,  0,  0]],

                    [[ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  1,  0],
                    [ 0,  1, -3,  0,  0],
                    [ 0,  2,  1,  0,  0],
                    [ 0,  0,  0,  0,  0]],

                    [[ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  0,  0],
                    [-1,  3, -3,  1,  0],
                    [ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  0,  0]]], dtype=torch.float32
)/float(255)

filter_bank_c = torch.tensor(
                    [[[ 0,  0,  0,  0,  0],
                    [ 0,  0,  1,  0,  0],
                    [ 0,  0, -2,  0,  0],
                    [ 0,  0,  1,  0,  0],
                    [ 0,  0,  0,  0,  0]],

                    [[ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  0,  0],
                    [ 0,  1, -2,  1,  0],
                    [ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  0,  0]],

                    [[ 0,  0,  0,  0,  0],
                    [ 0,  1,  0,  0,  0],
                    [ 0,  0, -2,  0,  0],
                    [ 0,  0,  0,  1,  0],
                    [ 0,  0,  0,  0,  0]],

                    [[ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  1,  0],
                    [ 0,  0, -2,  0,  0],
                    [ 0,  1,  0,  0,  0],
                    [ 0,  0,  0,  0,  0]]], dtype=torch.float32
)/float(255)

filter_bank_d = torch.tensor(
                    [[[ 0,  0,  0,  0,  0],
                    [ 0, -1,  2, -1,  0],
                    [ 0,  2, -4,  2,  0],
                    [ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  0,  0]],

                    [[ 0,  0,  0,  0,  0],
                    [ 0, -1,  2,  0,  0],
                    [ 0,  2, -4,  0,  0],
                    [ 0, -1,  2,  0,  0],
                    [ 0,  0,  0,  0,  0]],

                    [[ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  0,  0],
                    [ 0,  2, -4,  2,  0],
                    [ 0, -1,  2, -1,  0],
                    [ 0,  0,  0,  0,  0]],

                    [[ 0,  0,  0,  0,  0],
                    [ 0,  0,  2, -1,  0],
                    [ 0,  0, -4,  2,  0],
                    [ 0,  0,  2, -1,  0],
                    [ 0,  0,  0,  0,  0]]], dtype=torch.float32
)/float(255)

filter_bank_e = torch.tensor(
                [[[  1,   2,  -2,   2,   1],
                    [  2,  -6,   8,  -6,   2],
                    [ -2,   8, -12,   8,  -2],
                    [  0,   0,   0,   0,   0],
                    [  0,   0,   0,   0,   0]],

                [[  1,   2,  -2,   0,   0],
                    [  2,  -6,   8,   0,   0],
                    [ -2,   8, -12,   0,   0],
                    [  2,  -6,   8,   0,   0],
                    [  1,   2,  -2,   0,   0]],

                [[  0,   0,   0,   0,   0],
                    [  0,   0,   0,   0,   0],
                    [ -2,   8, -12,   8,  -2],
                    [  2,  -6,   8,  -6,   2],
                    [  1,   2,  -2,   2,   1]],

                [[  0,   0,  -2,   2,   1],
                    [  0,   0,   8,  -6,   2],
                    [  0,   0, -12,   8,  -2],
                    [  0,   0,   8,  -6,   2],
                    [  0,   0,  -2,   2,   1]]], dtype=torch.float32
)/float(255)

filter_bank_f = torch.tensor(
                    [[ 0,  0,  0,  0,  0],
                    [ 0,  -1,  2, -1,  0],
                    [ 0,  2,  -4,  2,  0],
                    [ 0,  -1,  2, -1,  0],
                    [ 0,  0,  0,  0,  0]], dtype=torch.float32
)/float(255)

filter_bank_g = torch.tensor(
                    [[ -1,   2,  -2,   2,  -1],
                    [  2,  -6,   8,  -6,   2],
                    [ -2,   8, -12,   8,  -2],
                    [  2,  -6,   8,  -6,   2],
                    [ -1,   2,  -2,   2,  -1]], dtype=torch.float32
)/float(255)

def apply_filter(img):
    tensor_a = torch.tensor(F.conv2d(img, filter_bank_a.unsqueeze(1), bias = None, stride = 1, padding = 0)).sum(dim =1, keepdim = True)
    tensor_b = torch.tensor(F.conv2d(img, filter_bank_b.unsqueeze(1), bias = None, stride = 1, padding = 0)).sum(dim =1, keepdim = True)
    tensor_c = torch.tensor(F.conv2d(img, filter_bank_c.unsqueeze(1), bias = None, stride = 1, padding = 0)).sum(dim =1, keepdim = True)
    tensor_d = torch.tensor(F.conv2d(img, filter_bank_d.unsqueeze(1), bias = None, stride = 1, padding = 0)).sum(dim =1, keepdim = True)
    tensor_e = torch.tensor(F.conv2d(img, filter_bank_e.unsqueeze(1), bias = None, stride = 1, padding = 0)).sum(dim =1, keepdim = True)
    tensor_f = torch.tensor(F.conv2d(img, filter_bank_f.unsqueeze(0).unsqueeze(0), bias = None, stride = 1, padding = 0)).sum(dim =1, keepdim= True)
    tensor_g = torch.tensor(F.conv2d(img, filter_bank_g.unsqueeze(0).unsqueeze(0), bias=None, stride=1, padding=0)).sum(dim =1, keepdim= True)
    tensor_img = (tensor_a/8.0 + tensor_b/8.0 + tensor_c/4.0 + tensor_d/4.0 + tensor_e/4.0 + tensor_f + tensor_g)
    median = tensor_img.median()
    threshold = median + 2/255.0
    noise_img = (tensor_img > threshold).float()
    return noise_img

if __name__ == "__main__":
    img = torch.randn(1, 1, 256, 256)
    print(apply_filter(img).size())