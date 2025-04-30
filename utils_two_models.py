import torch
import numpy as np
from collections import defaultdict
from PIL import Image

def deprocess_image(tensor):
    img = tensor * 0.5 + 0.5
    img = (img * 255).clamp(0, 255).byte().cpu().numpy()
    return img


def normalize(x):
    return x / (torch.sqrt(torch.mean(x ** 2)) + 1e-5)

def constraint_occl(grad, start_point, size):
    b, c, h, w = grad.shape
    x, y = start_point
    dx, dy = size
    mask = torch.zeros_like(grad)
    mask[:, :, y:y+dy, x:x+dx] = 1
    return grad * mask


def constraint_light(grad):
    return grad.clamp(-0.1, 0.1)


def constraint_black(grad, rect_shape=(6, 6)):
    b, c, h, w = grad.shape
    x = np.random.randint(0, w - rect_shape[1])
    y = np.random.randint(0, h - rect_shape[0])
    mask = torch.zeros_like(grad)
    patch = grad[:, :, y:y+rect_shape[0], x:x+rect_shape[1]]
    val = -torch.ones_like(patch) if patch.mean() < 0 else torch.ones_like(patch)
    mask[:, :, y:y+rect_shape[0], x:x+rect_shape[1]] = val
    return mask


def init_dict(model, model_layer_dict):
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            out_dim = getattr(module, 'out_channels', None) or getattr(module, 'out_features', None)
            if out_dim:
                model_layer_dict[name] = torch.zeros(out_dim)


def init_coverage_tables(model1, model2):
    model_layer_dict1 = dict()
    model_layer_dict2 = dict()
    init_dict(model1, model_layer_dict1)
    init_dict(model2, model_layer_dict2)
    return model_layer_dict1, model_layer_dict2

def neuron_to_cover(model_layer_dict):
    for name, tensor in model_layer_dict.items():
        if (tensor == 0).any():
            idx = (tensor == 0).nonzero(as_tuple=True)[0][0].item()
            return name, idx
    name = list(model_layer_dict.keys())[0]
    return name, 0


def neuron_covered(layer_dict):
    total = sum(v.numel() for v in layer_dict.values())
    covered = sum((v > 0).sum().item() for v in layer_dict.values())
    percentage = covered / total * 100 if total > 0 else 0
    return covered, total, percentage


def update_coverage(img_tensor, model, layer_dict, threshold):
    model.eval()
    handles = []

    def make_hook(name):
        def hook_fn(module, inp, out):
            act = out.detach().cpu()
            if act.dim() == 4:
                act = act.mean(dim=(2, 3))  # for Conv layers
            elif act.dim() == 2:
                pass  # Linear layer
            mask = (act > threshold).int()
            layer_dict[name] = (layer_dict[name] + mask[0]).clamp(0, 1)
        return hook_fn

    for name, module in model.named_modules():
        if name in layer_dict:
            handles.append(module.register_forward_hook(make_hook(name)))

    _ = model(img_tensor)

    for h in handles:
        h.remove()

def diverged(pred1, pred2):
    return not (pred1 == pred2)
