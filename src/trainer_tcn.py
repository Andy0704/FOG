import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import BinaryF1Score, BinaryStatScores

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        targets = targets.float().view(-1)
        inputs = inputs.view(-1)
        bce_loss = self.bce(inputs, targets)
        p_t = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * bce_loss
        return focal_loss.mean()

class TCNTrainer:
    def __init__(self, model, device='cpu'):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.criterion = BinaryFocalLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
        self.history = {'train_loss': [], 'val_loss': [], 'epoch_f1' : [], 'epoch_fpr' : []}

        # 加入 torchmetrics 並移動到對應的裝置
        self.val_f1 = BinaryF1Score(threshold=0.4).to(self.device)
        self.val_stats = BinaryStatScores(threshold=0.4).to(self.device)

    def train(self, train_loader, val_loader, n_epochs=20):
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 5  # 設定我們最多能容忍連續幾個 Epoch 沒進步

        print(f"開始訓練，使用裝置: {self.device}")
        for epoch in range(n_epochs):
            self.model.train()
            train_loss = 0.0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                
            val_loss , epoch_f1, epoch_fpr = self._validate(val_loader)
            avg_train_loss = train_loss / len(train_loader)

            # 存入 history   
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['epoch_f1'].append(epoch_f1)
            self.history['epoch_fpr'].append(epoch_fpr)
            print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | F1: {epoch_f1:.4f} | FPR: {epoch_fpr:.4f}")

            #Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
                print("發現最佳模型，已儲存！")
            else:
                patience_counter += 1
                print(f"Val Loss 未進步，容忍度: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("觸發 Early Stopping，提早結束訓練！")
                    break

            print("🔍 正在載入最佳模型權重進行 ROC 分析...")


    def _validate(self, loader):
        self.model.eval()
        val_loss = 0.0

        # 清空指標狀態
        self.val_f1.reset()
        self.val_stats.reset()
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                # 1. 將輸出轉換為機率，並使用 .view(-1) 壓扁成一維陣列
                probs = torch.sigmoid(outputs).view(-1)
                # 2. 更新指標 (torchmetrics 預設會以 0.5 為閥值進行判斷)
                self.val_f1.update(probs.view(-1), labels.view(-1).int())
                self.val_stats.update(probs.view(-1), labels.view(-1).int())

        # 迴圈結束後，計算整個 Epoch 的最終結果
        epoch_f1 = self.val_f1.compute().item()
        tp, fp, tn, fn, _ = self.val_stats.compute()
        print(f"DEBUG -> TP: {tp.item()}, FP: {fp.item()}, TN: {tn.item()}, FN: {fn.item()}")

        # 計算 FPR (加上 1e-6 是為了避免分母剛好為 0 導致程式崩潰)
        epoch_fpr = fp.item() / (tn.item() + fp.item() + 1e-6)

        # 將加總的 loss 除以批次數量，得到真正的平均 Val Loss
        val_loss = val_loss / len(loader)
        
        # 同時回傳三個指標
        return val_loss, epoch_f1, epoch_fpr