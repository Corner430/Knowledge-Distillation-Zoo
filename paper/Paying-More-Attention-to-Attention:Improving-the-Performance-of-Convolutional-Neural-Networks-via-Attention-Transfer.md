- **`AT`的目的是让student的attention map和teacher的attention map尽量一致**
- 多少层并不决定上限，多少 parameters 才决定上限
- 底盘要稳，才可以拟合 spatial map
- 作者做了一个隐形假设，就是 隐藏层激活之后的神经元的绝对值之间是很大联系的

![20240401214918](https://cdn.jsdelivr.net/gh/Corner430/Picture1/images/20240401214918.png)

![20240401214957](https://cdn.jsdelivr.net/gh/Corner430/Picture1/images/20240401214957.png)

```python
class AT(nn.Module):
    def __init__(self, p):
        super(AT, self).__init__()
        self.p = p

    def forward(self, fm_s, fm_t):
        loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))

        return loss

    def attention_map(self, fm, eps=1e-6):
        # 绝对值的 p 次方
        am = torch.pow(torch.abs(fm), self.p)
        # 对张量 am 沿着第二个维度（通道）进行求和
        am = torch.sum(am, dim=1, keepdim=True)
        norm = torch.norm(am, dim=(2, 3), keepdim=True)
        # 将张量 am 的每个元素除以张量 norm 的对应元素加上一个小的常数 eps
        am = torch.div(am, norm + eps)
        return am
```