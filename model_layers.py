import load_resnet as lr
import torch


resnet18_layernames = [
    'layer1.0',
    'layer1.1',
    'layer2.0',
    'layer2.1',
    'layer3.0',
    'layer3.1',
    'layer4.0',
    'layer4.1',
    'fc',
]


resnet50_layernames = [
    'layer1.0',
    'layer1.1',
    'layer1.2',
    'layer2.0',
    'layer2.1',
    'layer2.2',
    'layer2.3',
    'layer3.0',
    'layer3.1',
    'layer3.2',
    'layer3.3',
    'layer3.4',
    'layer3.5',
    'layer4.0',
    'layer4.1',
    'layer4.2',
    'fc',
]


def print_model_architecture(model, indent=0):
    for name, module in model.named_children():
        print('  ' * indent + f"({name}): {module.__class__.__name__}")
        if len(list(module.children())) > 0:
            print_model_architecture(module, indent + 1)
        else:
            print('  ' * (indent + 1) + str(module))


if __name__ == "__main__":
    model = lr.load_resnet("resnet18", "breast_mnist")
    print_model_architecture(model)
