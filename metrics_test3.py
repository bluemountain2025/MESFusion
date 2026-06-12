import os
import torch
import pandas as pd
from metric import read_image, calc

# Set input folder paths
ir_folder = "/root/autodl-tmp/guangzi/data/testdata/TNO/ir"
vis_folder = "/root/autodl-tmp/guangzi/data/testdata/TNO/vi"
fuse_folder = "/root/autodl-tmp/DAFNet/test_result/TNO"

# Path for saving results
results_path = "/root/autodl-tmp/DAFNet/DAF_TNO.xlsx"

# Remove the existing results file so a new one can be created
if os.path.exists(results_path):
    os.remove(results_path)

# Collect image filenames from each folder
ir_files = sorted(os.listdir(ir_folder))
vis_files = sorted(os.listdir(vis_folder))
fuse_files = sorted(os.listdir(fuse_folder))

# Ensure the file counts match
assert len(ir_files) == len(vis_files) == len(fuse_files), "IR, VIS, and FUSE folder must have the same number of files"

# Initialize a dictionary to store the results
metrics_results = {
    'filename': [],
    'SF': [],
    'EN': [],
    'AG': [],
    'SD': [],
    'CE': [],
    'CC': [],
    'SCD': [],
    'MSE': [],
    'MI': [],
    'PSNR': [],
    'Qabf': [],
    'VIF': [],
    'MS-SSIM': []
}

# Iterate over each image and compute the quality metrics
for ir_file, vis_file, fuse_file in zip(ir_files, vis_files, fuse_files):
    ir_path = os.path.join(ir_folder, ir_file)
    vis_path = os.path.join(vis_folder, vis_file)
    fuse_path = os.path.join(fuse_folder, fuse_file)
    
    # Read the images
    ir = read_image(ir_path)
    vis = read_image(vis_path)
    fuse = read_image(fuse_path)

    # Ensure the image shapes match
    assert ir.shape == vis.shape == fuse.shape, f"Image shapes do not match for files: {ir_file}, {vis_file}, {fuse_file}"

    # Compute the quality evaluation metrics
    metrics = calc(vis, ir, fuse)
    
    # Save the results
    metrics_results['filename'].append(ir_file)
    for key, value in metrics.items():
        metrics_results[key].append(value)

    print(f"Processed {ir_file}")

# Save the results to an Excel file
df = pd.DataFrame(metrics_results)
df.to_excel(results_path, index=False)

print(f"Results saved to {results_path}")
