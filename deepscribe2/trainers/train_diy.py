import torch
from torch.utils.data import DataLoader
from deepscribe2.datasets.dataset import DeepScribeDataset, collate_retinanet
from torchvision.models.resnet import resnet50
from torchvision.models.detection.retinanet import RetinaNet
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor

fold0_file = "/local/ecw/data_nov_2021_fixednumerals/folds/fold0.json"
fold1_file = "/local/ecw/data_nov_2021_fixednumerals/folds/fold1.json"

imgs = "/local/ecw/data_nov_2021_fixednumerals/cropped"

n_epochs = 1

loader = DataLoader(
    DeepScribeDataset(fold0_file, imgs, box_only=True),
    batch_size=5,
    collate_fn=collate_retinanet,
    num_workers=12,
)
val_loader = DataLoader(
    DeepScribeDataset(fold1_file, imgs, box_only=True),
    batch_size=5,
    collate_fn=collate_retinanet,
    num_workers=12,
)

# num_classes is INCLUDING the background?


backbone = _resnet_fpn_extractor(
    resnet50(), 5, returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(256, 256)
)

model = RetinaNet(
    backbone,
    num_classes=2,
    min_size=500,
    detections_per_img=500,
)


model = model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

model.train()

for epoch in range(n_epochs):
    for batch_num, (images, targets) in enumerate(loader):
        losses = model(
            [img.cuda() for img in images],
            [{key: val.cuda() for key, val in targ.items()} for targ in targets],
        )
        total_loss = sum(loss for loss in losses.values())

        print(losses, total_loss)
        # for param in model.parameters():
        #     print(param)
        #     break
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

# model.eval()

# for batch_num, (images, targets) in enumerate(loader):
#     preds = model([img.cuda() for img in images])

#     print(preds)

#     break
