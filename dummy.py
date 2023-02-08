import torch
import torch.nn.functional as F

input_mask = torch.load("input_mask_cpu.pt")

input_mask = input_mask.cuda()

x_shape = torch.Size([1, 4096])

new_mask = F.pad(input_mask, (0, x_shape[1] - input_mask.shape[1]), value=False)