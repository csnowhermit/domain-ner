import torch
import torch.nn as nn

import config


tmp = torch.zeros([config.batch_size, 1], dtype=torch.long)    # [2, 1]
print(tmp.shape)

tags = torch.zeros((2, 11), dtype=torch.long)    # [2, 11]
print(tags.shape)

tags = torch.cat([tmp, tags], dim=0)    # [2, 12]
print(tags.shape)

tags[:, 0] = 3
print(tags)