import os
import argparse
import random
import torch
import torch.nn as nn
import numpy as np
import detectors
from PIL import Image
import timm
from torchvision import datasets, transforms
from utils_two_models import *
import torch.nn.functional as F

# set seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# read the parameter
parser = argparse.ArgumentParser(description='Main function for difference-inducing input generation in CIFAR-10')
parser.add_argument('transformation', choices=['light', 'occl', 'blackout', 'no'])
parser.add_argument('weight_diff', type=float)
parser.add_argument('weight_nc', type=float)
parser.add_argument('step', type=float)
parser.add_argument('seeds', type=int)
parser.add_argument('grad_iterations', type=int)
parser.add_argument('threshold', type=float)
parser.add_argument('-t', '--target_model', choices=[0, 1, 2], default=0, type=int)
parser.add_argument('-sp', '--start_point', nargs=2, type=int, default=(0, 0))
parser.add_argument('-occl_size', '--occlusion_size', nargs=2, type=int, default=(10, 10))
args = parser.parse_args()

args.start_point = tuple(args.start_point)
args.occlusion_size = tuple(args.occlusion_size)
grad_iter = f"{args.grad_iterations}"
exp_name = f"{args.transformation}_wd{args.weight_diff}_wnc{args.weight_nc}_step{args.step}_t{args.target_model}"
os.makedirs(f"generated_inputs/grad_iter_{grad_iter}/{exp_name}", exist_ok=True)
log_file = open(f"generated_inputs/grad_iter_{grad_iter}/{exp_name}/log.txt", "w")


def save_image(image, file_path):
    image = Image.fromarray(image)
    image.save(file_path)

class LinearClassifier(nn.Module):
    def __init__(self, encoder, num_classes=10):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(encoder.num_features, num_classes)

    def forward(self, x):
        with torch.no_grad():  # encoder는 forward만 사용
            feats = self.encoder.forward_features(x)
        if feats.ndim == 4:  # (B, C, H, W)
            feats = F.adaptive_avg_pool2d(feats, (1, 1)).view(feats.size(0), -1)  # (B, C)
        return self.fc(feats)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.201))
])

testset = datasets.CIFAR10(root="./dataset", train=False, download=True, transform=transform)

# models are from https://huggingface.co/edadaltocg
model1 = timm.create_model("resnet50_cifar10", pretrained=True, num_classes=10).eval().cuda() # original training 방식
backbone_simclr = timm.create_model("resnet50_simclr_cifar10", pretrained=True) # self-supervised learning 방식
model2 = LinearClassifier(backbone_simclr).eval().cuda() # classifier없이 hf에 업로드 돼있어서 따로 classifier를 붙여 finetuning해줌. hf에 기재된 test acc를 train early stopping 기준으로 삼음.
model2.fc.load_state_dict(torch.load("linear_fc_simclr.pth", weights_only=True))

model_layer_dict1, model_layer_dict2 = init_coverage_tables(model1, model2)
activation_dict = {}

def register_hooks_multi_layers(model, prefix, activation_dict, layer_names):
    for layer_name in layer_names:
        module = model.get_submodule(layer_name)
        key = f"{prefix}_{layer_name}"
        module.register_forward_hook(get_activation_hook(key, activation_dict))

def get_activation_hook(name, activation_dict):
    def hook(model, input, output):
        activation_dict[name] = output
    return hook

# model.named_modules()를 보았을 때 layer가 4개, 그 중 act2가 relu여서 다음과 같이 설정하였음.

layer_names_model1 = [
    "layer1.0.act2",
    "layer2.0.act2",
    "layer3.0.act2",
    "layer4.0.act2",
]

layer_names_model2 = [
    "encoder.layer1.0.act2",
    "encoder.layer2.0.act2",
    "encoder.layer3.0.act2",
    "encoder.layer4.0.act2",
]

activation_dict = {}
register_hooks_multi_layers(model1, "model1", activation_dict, layer_names_model1)
register_hooks_multi_layers(model2, "model2", activation_dict, layer_names_model2)

def log_neuron_coverage(seed_idx, nc1, nc2, prefix=""):
    avg_nc = (nc1[0] + nc2[0]) / (nc1[1] + nc2[1])
    log_message = (
        f"[Seed {seed_idx}] {prefix}Neuron Coverage - "
        f"model1: {nc1[2]:.3f}%, model2: {nc2[2]:.3f}%, avg: {avg_nc * 100:.3f}%\n"
    )
    log_file.write(log_message)
                                                                                                    
for seed_idx in range(args.seeds):
    random.seed(seed_idx)
    np.random.seed(seed_idx)
    torch.manual_seed(seed_idx)

    index = random.randint(0, len(testset) - 1)
    raw_img = testset.data[index]
    true_label = testset.targets[index]
    pil_img = Image.fromarray(raw_img)    
    input_tensor = transform(pil_img).unsqueeze(0).cuda().requires_grad_(True)
    gen_img = input_tensor.clone().detach().requires_grad_(True)
    orig_img = input_tensor.clone().detach()

    logits1 = model1(gen_img)
    logits2 = model2(gen_img)
    label1 = torch.argmax(logits1, dim=1).item()
    label2 = torch.argmax(logits2, dim=1).item()

    if label1 != label2:
        log_file.write(f"[Seed {seed_idx}] Already diverged: {label1}, {label2} | GT: {true_label}\n")
        update_coverage(gen_img, model1, model_layer_dict1, args.threshold)
        update_coverage(gen_img, model2, model_layer_dict2, args.threshold)

        nc1 = neuron_covered(model_layer_dict1)
        nc2 = neuron_covered(model_layer_dict2)
        log_neuron_coverage(seed_idx, nc1, nc2)

        save_image(deprocess_image(gen_img[0].permute(1, 2, 0)), 
                   f'generated_inputs/grad_iter_{grad_iter}/{exp_name}/already_differ_{seed_idx}_{label1}_{label2}.png')
        continue

    log_file.write(f"[Seed {seed_idx}] Same: {label1}, {label2} | GT: {true_label}\n")
    orig_label = true_label

    found_divergence = False
    for iters in range(args.grad_iterations):
        gen_img.requires_grad = True
        logits1 = model1(gen_img)
        logits2 = model2(gen_img)
        label_tensor = torch.tensor([orig_label]).cuda()

        if args.target_model == 0:
            loss1 = -args.weight_diff * F.cross_entropy(logits1, label_tensor)
            loss2 = F.cross_entropy(logits2, label_tensor)
        elif args.target_model == 1:
            loss1 = F.cross_entropy(logits1, label_tensor)
            loss2 = -args.weight_diff * F.cross_entropy(logits2, label_tensor)

        loss_nc1, loss_nc2 = 0, 0
        for name in layer_names_model1:
            key = f"model1_{name}"
            feat = activation_dict[key]
            if feat.ndim == 4:
                feat = feat.mean(dim=(2, 3))
            C = feat.shape[1]
            idx = random.randint(0, C - 1)
            loss_nc1 += torch.mean(feat[:, idx])

        for name in layer_names_model2:
            key = f"model2_{name}"
            feat = activation_dict[key]
            if feat.ndim == 4:
                feat = feat.mean(dim=(2, 3))
            C = feat.shape[1]
            idx = random.randint(0, C - 1)
            loss_nc2 += torch.mean(feat[:, idx])

        total_loss = (loss1 + loss2) + args.weight_nc * (loss_nc1 + loss_nc2)

        log_file.write(f"  [Loss] iter {iters}: loss1={loss1.item():.4f}, loss2={loss2.item():.4f}, "
                       f"nc1={loss_nc1.item():.4f}, nc2={loss_nc2.item():.4f}\n")

        gen_img.grad = None
        total_loss.backward()
        grad = gen_img.grad

        if args.transformation == 'light':
            grad = constraint_light(grad)
        elif args.transformation == 'occl':
            grad = constraint_occl(grad, args.start_point, args.occlusion_size)
        elif args.transformation == 'blackout':
            grad = constraint_black(grad)

        gen_img = (gen_img + args.step * grad).detach()
        
        pred1 = torch.argmax(model1(gen_img), dim=1).item()
        pred2 = torch.argmax(model2(gen_img), dim=1).item()

        if pred1 != pred2:
            log_file.write(f"[Seed {seed_idx}] Diverged at iter {iters}: {pred1}, {pred2} | GT: {true_label}\n")
            update_coverage(gen_img, model1, model_layer_dict1, args.threshold)
            update_coverage(gen_img, model2, model_layer_dict2, args.threshold)

            nc1 = neuron_covered(model_layer_dict1)
            nc2 = neuron_covered(model_layer_dict2)
            log_neuron_coverage(seed_idx, nc1, nc2)

            gen_img_np = deprocess_image(gen_img[0].permute(1, 2, 0))
            orig_img_np = deprocess_image(orig_img[0].permute(1, 2, 0))

            save_image(gen_img_np, 
                       f'generated_inputs/grad_iter_{grad_iter}/{exp_name}/{args.transformation}_{seed_idx}_{pred1}_{pred2}.png')
            save_image(orig_img_np, 
                       f'generated_inputs/grad_iter_{grad_iter}/{exp_name}/{args.transformation}_{seed_idx}_{pred1}_{pred2}_orig.png')
            found_divergence = True
            break

    if not found_divergence:
        update_coverage(gen_img, model1, model_layer_dict1, args.threshold)
        update_coverage(gen_img, model2, model_layer_dict2, args.threshold)

        nc1 = neuron_covered(model_layer_dict1)
        nc2 = neuron_covered(model_layer_dict2)
        log_neuron_coverage(seed_idx, nc1, nc2, prefix="Final ")

log_file.close()
