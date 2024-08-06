import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def pairwise_concatenate_and_score(tensors, mlp):
    num_tensors, tensor_size = tensors.size()

    # Expand tensors for pairwise concatenation
    expanded1 = tensors.unsqueeze(1).expand(-1, num_tensors, -1)
    expanded2 = tensors.unsqueeze(0).expand(num_tensors, -1, -1)

    # Concatenate the expanded tensors along the last dimension
    concatenated_matrix = torch.cat((expanded1, expanded2), dim=2)

    # Reshape concatenated_matrix to (num_tensors * num_tensors, 2 * tensor_size)
    concatenated_matrix = concatenated_matrix.view(-1, 2 * tensor_size)

    # 通过 MLP 计算注意力分数
    attention_scores = mlp(concatenated_matrix).view(num_tensors, num_tensors)

    return concatenated_matrix, attention_scores


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        input_dim = 1024  # 假设拼接后的向量维度为 1024 (512 + 512)
        hidden_dim = 512
        output_dim = 1
        self.fc_att = SimpleMLP(input_dim, hidden_dim, output_dim)

    def forward(self, f):
        concatenated_matrix, attention_scores = pairwise_concatenate_and_score(f, self.fc_att)

        # 将对角线上的元素设置为1
        num_tensors = attention_scores.size(0)
        mask = torch.eye(num_tensors, dtype=torch.bool, device=attention_scores.device)
        attention_scores[mask] = 1

        return concatenated_matrix, attention_scores


# 示例使用
seq = torch.randn((2485, 512))

model = Model()
concatenated_matrix, attention_scores = model(seq)

print("Concatenated Matrix:")
print(concatenated_matrix)
print("Attention Scores:")
print(attention_scores)
