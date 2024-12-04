# import torch
# import torch.nn as nn

# class SimilarityModule(nn.Module):
#     def __init__(self, k_d, num_k):
#         super(SimilarityModule, self).__init__()
#         # 可学习参数 k 和 a
#         self.k = nn.Parameter(torch.randn(num_k, k_d))  # (100, k_d)
#         self.a = nn.Parameter(torch.randn(num_k, k_d))  # (100, k_d)

#     def forward(self, q):
#         # q 的形状是 (b, l, d)
        
#         # 1. 计算 q 与 k 的相似度
#         # 需要松耦合扩展
#         similarity = torch.einsum('bld,kd->blk', q, self.k)  # (b, l, 100)
        
#         # 2. 计算与 a 的乘法产出
#         output = torch.einsum('blk,kd->bld', similarity, self.a)  # (b, l, d)
        
#         # 3. 对 output 求和得到标量 (对于 l 维的每个值求和)
#         # scalar_output = output.sum(dim=1)  # (b, d)
        
#         # 例如，如果想要另外聚合，结合 torch.mean 则可以得到( b, d)

#         return output

# # 测试
# if __name__ == "__main__":
#     b = 2  # batch size
#     l = 5  # sequence length
#     d = 10 # feature dimension
#     k_d = 10  # k 的维度
#     num_k = 100  # 100个 k

#     # 实例化模块
#     similarity_module = SimilarityModule(k_d, num_k)

#     # 随机生成 q
#     query = torch.randn(b, l, d)
#     v = torch.randn(num_k, d)

#     # 计算标量输出
#     scalar_output = similarity_module(query)
#     # xx = torch.einsum('blk,kd->bld', scalar_output, v)
#     breakpoint()
#     print("Scalar Output Shape:", scalar_output.shape)  # 应该为 (b, d)
#     # print("Scalar Output:", scalar_output)


import torch

# 创建两个一维张量组成的列表
tensor_list = [torch.tensor([3], dtype=torch.float), torch.tensor([4], dtype=torch.float)]

# 分别求每个张量的均值
breakpoint()
mean_list = [t.mean().item() for t in tensor_list]
# 此时mean_list为 [3.0, 4.0]，分别是两个张量的均值

# 对这些均值再求平均
final_mean = sum(mean_list) / len(mean_list)
print(final_mean)