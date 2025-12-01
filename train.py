import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, backbone, loss_fn, loader: DataLoader, num_classes, save_path,
                 lr=1e-4, device=None, patience=5):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = backbone.to(self.device)
        # self.rerank = rerank.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.loader = loader
        self.num_classes = num_classes
        self.save_path = save_path
        self.patience = patience

        # Learnable classifier for ArcFace
        self.W = nn.Parameter(torch.randn(backbone.reduction_dim, num_classes, device=self.device))
        nn.init.xavier_uniform_(self.W)

        self.optimizer = torch.optim.Adam(
            list(self.backbone.parameters())  + [self.W],
            lr=lr, weight_decay=1e-5
        )

        self.best_loss = float("inf")
        self.no_improve_count = 0

    def train_epoch(self):
        self.backbone.train()
        total_loss = 0.0

        for im_list, labels in self.loader:
            batch_size = len(im_list)

            imgs = torch.stack([torch.tensor(x, dtype=torch.float32) for x in im_list], dim=0).to(self.device)

            labels = torch.tensor(labels, dtype=torch.long, device=self.device)

            # Forward
            features = self.backbone(imgs)

            # Cosine similarity
            W_norm = F.normalize(self.W, dim=0)
            features_norm = F.normalize(features, dim=1)
            cosine = torch.matmul(features_norm, W_norm)

            # ArcFace loss
            loss = self.loss_fn(cosine, labels)

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.loader)
        return avg_loss

    def train(self, num_epochs=50):
        for epoch in range(num_epochs):
            avg_loss = self.train_epoch()
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

            # Early stopping check
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.no_improve_count = 0
                # Save best model
                torch.save({
                    "backbone": self.backbone.state_dict(),
                    "W": self.W,
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch
                }, self.save_path)
                print(f"Saved best model at epoch {epoch+1}")
            else:
                self.no_improve_count += 1
                if self.no_improve_count >= self.patience:
                    print(f"No improvement for {self.patience} epochs. Early stopping.")
                    break

if __name__ == "__main__": 
    import model.backbone.resnet as backbone
    from dataloader.test_loader import _construct_loader
    from model.loss.arcsoftmax import ArcLoss
    import torch
    from torch.utils.data import DataLoader

    # ----------------------------
    # Dataset / Loader
    # ----------------------------
    DATA_DIR = "data/datasets"
    dataset_name = "oxford5k"
    fn = "gnd_oxford.pkl"
    split = "db"  # database images for training
    scale_list = [1.0]  # single-scale training
    batch_size = 4

    loader = _construct_loader(DATA_DIR, dataset_name, fn, split, scale_list,
                               batch_size=batch_size, shuffle=True, drop_last=True)

    # ----------------------------
    # Model & Loss
    # ----------------------------
    resnet = backbone.ResNet(depth=50, reduction_dim=512, relup=0.01)
    loss_fn = ArcLoss(s=64.0, m=0.5)

    # Number of classes = number of DB images / landmarks
    # Here we prepare real labels from the dataset
    from dataloader.dataset import DataSet
    dataset = DataSet(DATA_DIR, dataset_name, fn, split, scale_list)
    num_classes = len(dataset._prepare_labels())

    # ----------------------------
    # Trainer
    # ----------------------------
    save_path = "./best_model.pt"
    trainer = Trainer(backbone=resnet, loss_fn=loss_fn,
                      loader=loader, num_classes=num_classes, save_path=save_path,
                      lr=1e-4, patience=5)

    # ----------------------------
    # Train
    # ----------------------------
    trainer.train(num_epochs=50)

