import config
import torch
import torch.fx

from utils import *

model = symbolic_trace(config.get_model(seed=39)).cuda(0)

optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
train_data = config.get_data()[1]

result_times = []
for iter in range(100):
    x, y = next(train_data)
    x = x.cuda(0)
    y = y.cuda(0)
    with measure_time(f"iteration {iter}") as wall_time:
        loss = model(x, y)
        print(f"loss {iter}:", loss.detach().cpu().numpy())
        loss.backward()
        optimizer.step()
    print(wall_time)
    result_times.append(wall_time.time)
    print("avg:", sum(result_times[-50:]) / len(result_times[-50:]))
