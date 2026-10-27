import argparse

import torch
from torch.utils.data import DataLoader

from model import UNetSegmentation
from tusimple_dataset import TuSimpleDataset
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net on TuSimple lanes")
    parser.add_argument("--data", default="TUSimple/train_set")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--lane-weight", type=float, default=10.0)
    parser.add_argument("--output", default="unet_tusimple.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TuSimpleDataset(args.data, image_size=(args.height, args.width))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )

    model = UNetSegmentation().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    class_weights = torch.tensor([1.0, args.lane_weight], device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            model_output = model(images)
            loss = criterion(model_output, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        print(f"epoch {epoch + 1}/{args.epochs} loss={total_loss / len(dataset):.4f}")

    torch.save(model.state_dict(), args.output)
    print(f"saved model to {args.output}")


if __name__ == "__main__":
    main()
