import os

def write_filenames_to_txt(folder_path: str, output_txt_path: str) -> None:
    try:
        # Get all filenames in the folder
        filenames = os.listdir(folder_path)
        
        # Write filenames to the text file line by line
        with open(output_txt_path, 'w') as txt_file:
            for filename in filenames:
                # Only write files, not subdirectories
                if os.path.isfile(os.path.join(folder_path, filename)):
                    txt_file.write(filename + '\n')
        print(f"文件名已写入到 {output_txt_path}")
    except Exception as e:
        print(f"发生错误: {e}")

# Example usage
write_filenames_to_txt('/root/autodl-tmp/data/testdata/CTMRI/ir', '/root/autodl-tmp/data/testdata/CTMRI/labels.txt')
