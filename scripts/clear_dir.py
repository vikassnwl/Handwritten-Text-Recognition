import sys
import os

dir_pth = sys.argv[1]

for content_name in os.listdir(dir_pth):
    content_path = f'{dir_pth}/{content_name}'
    if not os.path.isdir(content_path):
        os.remove(content_path)