import torch

print("PyTorch版本:", torch.__version__)
print("CUDA是否可用:", torch.cuda.is_available())
print("PyTorch内置CUDA版本:", torch.version.cuda)  # 如果为 None，则100%是CPU版本

if torch.cuda.is_available():
    print("CUDA设备数量:", torch.cuda.device_count())
    print("当前CUDA设备:", torch.cuda.current_device())
    print("CUDA设备名称:", torch.cuda.get_device_name(0))
else:
    print("CUDA不可用，请检查环境配置。")