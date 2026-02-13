import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HANConv, global_mean_pool

class HAN_Detector(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, metadata):
        super().__init__()
        # HAN 层: 自动处理异构图的层级注意力
        self.han_conv = HANConv(in_channels, hidden_channels, heads=4, dropout=0.2, metadata=metadata)
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, batch_dict):
        # 1. HAN 卷积，获取节点特征
        out_dict = self.han_conv(x_dict, edge_index_dict)
        tx_feature = out_dict['transaction']
        
        # 2. 全局池化：将同一个图里的所有 Transaction 节点特征取平均，代表这张图
        # batch_dict['transaction'] 告诉模型哪些节点属于哪张图
        graph_embedding = global_mean_pool(tx_feature, batch_dict['transaction'])
        
        # 3. 分类输出 (未经过 Sigmoid，输出 Logits)
        return self.lin(graph_embedding)