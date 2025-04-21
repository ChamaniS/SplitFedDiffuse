import torch
from medsegbench import FHPsAOPMSBench
train_dataset = FHPsAOPMSBench(split="train", download=True)
