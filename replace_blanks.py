import os

def replace_spaces_in_txt_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    new_content = content.replace(' ', ',')
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"已处理文件: {file_path}")
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {e}")


if __name__ == "__main__":
    target_directory = 'dataset'
    replace_spaces_in_txt_files(target_directory)
    