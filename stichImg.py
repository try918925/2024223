import time
import threading
from new_util import stichAll, crop_img, estimate_translation_orb


class stiching_img(object):
    def __init__(self) -> None:
        self.newLst = []
        self.offLst = []
        self.newImg = []
        self.crop = crop_img()

    # 直接存全尺寸的frame进lst就行
    def stiching(self, img_list, v):
        # v代表摄像头方位：top,right,left
        for img in img_list:
            img_crop = self.crop.crop_images(img, v)
            self.newImg.append(img_crop)
            self.newLst.append(img_crop)
            if len(self.newLst) == 2:
                img1 = self.newLst[0]
                img2 = self.newLst[1]
                self.newLst.clear()
                offset = estimate_translation_orb(img1, img2, v)
                self.offLst.append(offset)
                self.newLst.append(img2)

        result = stichAll(self.newImg, self.offLst, v)
        # self.newLst.clear()
        # self.offLst.clear()
        # self.newImg.clear()

        return result



class stiching_no_crop(object):
    def __init__(self) -> None:
        self.newLst = []
        self.offLst = []
        self.newImg = []
        self.crop = crop_img()

    # 直接存全尺寸的frame进lst就行
    def stiching(self, img_list, v):
        # v代表摄像头方位：top,right,left
        for img in img_list:
            self.newImg.append(img)
            self.newLst.append(img)
            if len(self.newLst) == 2:
                img1 = self.newLst[0]
                img2 = self.newLst[1]
                self.newLst.clear()
                offset = estimate_translation_orb(img1, img2, v)
                self.offLst.append(offset)
                self.newLst.append(img2)

        result = stichAll(self.newImg, self.offLst, v)
        # self.newLst.clear()
        # self.offLst.clear()
        # self.newImg.clear()

        return result


class stiching_batches(object):
    def __init__(self) -> None:
        self.newLst = []
        self.offLst = []
        self.newImg = []
        self.crop = crop_img()

    # 直接存全尺寸的frame进lst就行
    def stiching(self, img_list, v, status):
        # v代表摄像头方位：top,right,left

        if self.newImg and img_list:
            img1 = self.newImg[-1]
            img2 = self.crop.crop_images(img_list[0], v)
            offset = estimate_translation_orb(img1, img2, v)
            self.offLst.append(offset)

        start_time = time.time()
        for index, img in enumerate(img_list):
            img_crop = self.crop.crop_images(img, v)
            self.newImg.append(img_crop)
            self.newLst.append(img_crop)
            if len(self.newLst) == 2:
                img1 = self.newLst[0]
                img2 = self.newLst[1]
                self.newLst.clear()
                offset = estimate_translation_orb(img1, img2, v)
                self.offLst.append(offset)
                self.newLst.append(img2)

        print(f"{v}:传图数量{len(img_list)},拼图时间:{time.time() - start_time}")
        if status:
            result = False
            if self.newImg and self.offLst:
                print("self.newImg:", len(self.newImg), "self.offLst:", len(self.offLst))
                start_time = time.time()
                result = stichAll(self.newImg, self.offLst, v)
                print("stichAll(self.newImg, self.offLst, v):", time.time() - start_time)
            self.newLst.clear()
            self.offLst.clear()
            self.newImg.clear()
            return result


class stiching_thread(object):
    def __init__(self) -> None:
        self.crop = crop_img()
        self.all_data = {}

    # 直接存全尺寸的frame进lst就行
    def stiching(self, img_list, v, thread_num, status):
        # v代表摄像头方位：top,right,left
        if img_list:
            newLst = []
            offLst = []
            newImg = []
            start_time = time.time()
            for img in img_list:
                img_crop = self.crop.crop_images(img, v)
                newImg.append(img_crop)
                newLst.append(img_crop)
                if len(newLst) == 2:
                    img1 = newLst[0]
                    img2 = newLst[1]
                    newLst.clear()
                    offset = estimate_translation_orb(img1, img2, v)
                    offLst.append(offset)
                    newLst.append(img2)
            self.all_data[thread_num] = {"newLst": newLst, "offLst": offLst, "newImg": newImg}
            print(f"{v}:传图数量{len(img_list)},拼图时间:{time.time() - start_time}")

        if status:
            result = False
            if self.all_data:
                sorted_dict = {key: self.all_data[key] for key in sorted(self.all_data.keys())}
                for key, value in sorted_dict:
                    pass

                print("self.newImg:", len(newImg), "self.offLst:", len(offLst))
                start_time = time.time()
                result = stichAll(newImg, offLst, v)
                print("stichAll(self.newImg, self.offLst, v):", time.time() - start_time)
            self.newLst.clear()
            self.offLst.clear()
            self.newImg.clear()
            return result


class stiching_distribution(object):
    def __init__(self) -> None:
        self.lock = threading.Lock()  # 初始化锁
        self.crop = crop_img()
        self.all_data = {}

    # 直接存全尺寸的frame进lst就行
    def stiching(self, img_list, v, num, status):
        # v代表摄像头方位：top,right,left
        with self.lock:  # 确保这里加锁
            self.all_data[num] = self.crop_images(img_list, v)
            # print(f"线程号为:{num}裁剪完成", )
            if status:
                for _ in range(200):
                    # 按 key 从小到大排序
                    sorted_keys = sorted(self.all_data.keys())
                    # 检查相邻的 key 是否都相差 1
                    all_adjacent_diff_one = all(
                        sorted_keys[i] - sorted_keys[i - 1] == 1 for i in range(1, len(sorted_keys)))
                    # 如果相邻 key 差值都为 1，则合并列表
                    if all_adjacent_diff_one:
                        merged_list = []
                        for key in sorted_keys:
                            merged_list += self.all_data[key]
                        result = self.stichall_images(merged_list, v)
                        return result, True
                    time.sleep(0.5)
            else:
                return False, False

    def crop_images(self, img_list, v):
        img_crop_list = []
        if img_list:
            for img in img_list:
                img_crop = self.crop.crop_images(img, v)
                img_crop_list.append(img_crop)
        return img_crop_list

    def stichall_images(self, img_list, v):
        newLst = []
        offLst = []
        newImg = []
        for img in img_list:
            newImg.append(img)
            newLst.append(img)
            if len(newLst) == 2:
                img1 = newLst[0]
                img2 = newLst[1]
                newLst.clear()
                offset = estimate_translation_orb(img1, img2, v)
                offLst.append(offset)
                newLst.append(img2)
        result = stichAll(newImg, offLst, v)
        self.all_data.clear()
        return result
