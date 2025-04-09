import torch
from medsegbench import BkaiIghMSBench
train_dataset = BkaiIghMSBench(split="train", download=True)
