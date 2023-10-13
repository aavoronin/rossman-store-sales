# https://www.youtube.com/watch?v=r7Am-ZGMef8

import GPUtil
from tabulate import tabulate


def check_GPU():
    import torch
    import os
    print(os.environ['CUDA_PATH'])
    print(os.environ['CUDA_HOME'])
    print(os.environ['PATH'])
    print(torch.backends.cudnn.enabled)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using", device, "device")

    def gpu_info():
        gpus = GPUtil.getGPUs()
        gpus_list = []

        for gpu in gpus:
            gpu_id = gpu.id
            gpu_name = gpu.name
            gpu_load = f'{gpu.load * 100}%'
            gpu_free_memory = f'{gpu.memoryFree}MB'
            gpu_used_memory = f'{gpu.memoryUsed}MB'
            gpu_total_memoru = f'{gpu.memoryTotal}MB'
            gpu_temp = f'{gpu.temperature}'

            gpus_list.append((
                gpu_id,
                gpu_name,
                gpu_load,
                gpu_free_memory,
                gpu_used_memory,
                gpu_total_memoru,
                gpu_temp
            ))
        return str(tabulate(
            gpus_list,
            headers=(
                'id',
                'name',
                'load',
                'free memory',
                'used memory',
                'total memory',
                'temperature'
            ),
            tablefmt='pretty'
        )
        )

    print(gpu_info())
