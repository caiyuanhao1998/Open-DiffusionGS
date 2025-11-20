# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).
import os
import json
import torch
from tqdm import tqdm
from diffusionGS.utils.losses import MetricComputer
import argparse

def compute_metrics(
    psnr_path: str,  # 关键字参数：原result_dir，存储.pt文件的目录
    chunk: int = 8   # 关键字参数：分块大小，默认8
):
    # 初始化MetricComputer
    metric_computer = MetricComputer()
    
    # 加载.pt文件数据
    all_image_results = []
    all_gts = []
    all_path_names = os.listdir(psnr_path)
    for path_name in tqdm(all_path_names):
        if path_name.endswith('.pt'):
            result_pkg = torch.load(os.path.join(psnr_path, path_name))
            all_image_results.append(result_pkg['render_images'])
            all_gts.append(result_pkg['image'])

    all_image_results = torch.stack(all_image_results)
    all_gts = torch.stack(all_gts)

    # 计算指标并收集结果
    all_psnr = []
    all_ssim = []
    all_lpips = []
    metric_computer.lpips_loss_module.cuda()
    for i in tqdm(range(0, len(all_image_results), chunk)):
        chunk_image_results = all_image_results[i:i+chunk]
        chunk_gts = all_gts[i:i+chunk]
        psnr, ssim, lpips = metric_computer(chunk_image_results.cuda(), chunk_gts.cuda())
        all_psnr.append(psnr)
        all_ssim.append(ssim)
        all_lpips.append(lpips)

    # 计算平均值并打印
    avg_psnr = torch.cat(all_psnr,dim=0).mean()
    avg_ssim = torch.cat(all_ssim,dim=0).mean()
    avg_lpips = torch.cat(all_lpips,dim=0).mean()
    print(f'psnr: {avg_psnr}, ssim: {avg_ssim}, lpips: {avg_lpips}')

    # 保存结果到JSON
    result_dump_file = os.path.join(psnr_path, 'eval_result.json')
    result_json = {
        'psnr': avg_psnr.item(),
        'ssim': avg_ssim.item(),
        'lpips': avg_lpips.item(),
    }
    with open(result_dump_file, 'w') as f:
        json.dump(result_json, f, indent=4)


if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='计算并保存图像指标（PSNR/SSIM/LPIPS）')
    
    # 添加命令行参数
    parser.add_argument(
        '--path', 
        type=str, 
        required=True, 
        help='存储.pt结果文件的目录路径'
    )
    parser.add_argument(
        '--chunk', 
        type=int, 
        default=8, 
        help='分块处理大小（默认值为8）'
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用函数，传递解析后的参数
    compute_metrics(psnr_path=args.path, chunk=args.chunk)