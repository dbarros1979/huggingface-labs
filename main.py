from transformers import pipeline;
import torch

print(pipeline('sentiment-analysis')('we love you'))

x = torch.rand(5, 3)
print(x)