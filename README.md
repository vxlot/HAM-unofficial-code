# HAM unofficial code
Hybrid attention module in deep convolutional neural networks for image classification.

![](https://raw.githubusercontent.com/vxlot/img_host/main/PicGo/ham.webp)

```python
!git clone https://github.com/vxlot/HAM-unofficial-code.git
%cd HAM

from ham import HAM
import torch


batch_size, channels, height, width = 1, 128, 64, 64
# channel_groups = channels // 2

F = torch.randn(batch_size, channels, height, width)
print('F',F.shape)

ham_attn = HAM(channels=channels, channel_groups=channels//2)
F = ham_attn(F)
F.shape
```
