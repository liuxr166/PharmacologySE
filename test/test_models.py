import torch
import torch_npu
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# 定义 BiLSTM 编码器模型
class BiLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super(BiLSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                              bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.bilstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# 示例参数
input_size = 10
hidden_size = 20
output_size = 5

# 创建模型实例并移动到设备（假设设备为 Ascend AI 处理器）
device = torch.device("npu" if torch_npu.npu.is_available() else "cpu")
model = BiLSTMEncoder(input_size, hidden_size, output_size).to(device)

# 创建数据集和数据加载器
batch_size = 10
input_tensor = torch.randn(100, 7, input_size)  # 示例数据
dataset = TensorDataset(input_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 优化训练循环以减少内存使用
optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()

for epoch in range(10):
    for batch in dataloader:
        batch = batch[0].to(device)
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, torch.randn(batch_size, output_size).to(device))
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch + 1}/10], Loss: {loss.item():.4f}')

# 确保数据传输和模型计算在设备上同步
torch_npu.npu.synchronize(device)
