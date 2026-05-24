import torch

print("torch_version=" + str(torch.__version__))
print("cuda_available=" + str(torch.cuda.is_available()))
print("cuda_version=" + str(torch.version.cuda))
if torch.cuda.is_available():
    print("device=" + torch.cuda.get_device_name(0))
else:
    print("device=None")
