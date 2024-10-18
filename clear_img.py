import os
import glob
import time


def get_all_images(directory):
    return glob.glob(os.path.join(directory, '*.jpg'))

def delete_old_images(directory, keep_latest=20):
    images = get_all_images(directory)
    if len(images) > keep_latest:
        images.sort(key=os.path.getmtime, reverse=True)
        old_images = images[keep_latest:]
        for img in old_images:
            os.remove(img)

def clean_all_directories(base_directory, keep_latest=20):
    for root, dirs, files in os.walk(base_directory):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            delete_old_images(dir_path, keep_latest)


if __name__ == '__main__':

    # 基础目录，替换为你的my_test_img目录路径
    base_directory = 'my_test_img'
    # 清理所有子目录，只保留最新的20张图片
    while True:
        try:
            clean_all_directories(base_directory, keep_latest=0)
            print("------完成--------------------")
        except Exception as error:
            print("error:",error)
        # time.sleep(2)
