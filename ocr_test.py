import infer_det_rec as det_ocr
import cv2
import os
import numpy as np

def calc_M(v):

    # v 表示当前图来源
    if v == 'right':
        trapezoid = np.array([[497, 146], [1587, 396], [457, 1321], [1551, 1114]], dtype=np.float32)
        rectangle = np.array([[497, 146], [1587, 146], [497, 1321], [1587, 1321]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(trapezoid, rectangle)

    elif v == 'left':
        trapezoid = np.array([[871, 367], [2140, 103], [875, 1182], [2136, 1426]], dtype=np.float32)
        rectangle = np.array([[871, 103], [2140, 103], [871, 1426], [2140, 1426]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(trapezoid, rectangle)

    elif v == 'top':
        trapezoid = np.array([[119, 289], [1091, 276], [120, 414], [1090, 418]], dtype=np.float32)  # 左上、右上、左下、右下
        rectangle = np.array([[119, 289], [1091, 289], [119, 414], [1091, 414]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(trapezoid, rectangle)

    return M


def run(video_file, rotate_tag):
    ocr_img_list = []

    config_dict = {
        "ocr_det_config": "./config/det/my_det_r50_db++_td_tr.yml",
        "ocr_rec_config": "./config/rec/my_en_PP-OCRv3_rec.yml"
    }
    my_ocr_process = det_ocr.OCR_process(config_dict)

    # 打开视频文件
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print("错误：无法打开视频文件。")
        exit()
    count = 0
    results_lst = []
    M = calc_M(rotate_tag)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            count += 1
            if rotate_tag == 'right':
                frame = cv2.warpPerspective(frame, M, (2688, 1520))
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotate_tag == 'left':
                frame = cv2.warpPerspective(frame, M, (2688, 1520))
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # img = crop_images(frame, "right")
            ocr_img_list.append(frame)
            # print(frame.shape)
            results = my_ocr_process.process_imgs(ocr_img_list)
            # if results is not None:
            #     results_lst.append(results)

            result1, result2 = results
            if result1 != ' ' and result2 != ' ':
                results_lst.append(results)
            #     print(results)
            ocr_img_list.clear()
        else:
            break
    
    # results = my_ocr_process.process_imgs_2(ocr_img_list)
    # print(results)
    # 释放视频流对象和写入对象
    cap.release()
    
    # results = my_ocr_process.process_imgs(ocr_img_list, rotate_tag)
    # print(results)

    return results_lst

def get_save_idx(score_lst):
    tmp_lst = []
    idx = 0

    max_length = max(len(sublist) for sublist in score_lst)
    print(max_length)
    for threshes in score_lst:
        if len(threshes) > 0 and len(threshes) == max_length:
            avg = sum(threshes)/len(threshes)
            tmp_lst.append(avg)
        else:
            tmp_lst.append(0)
    
    if len(tmp_lst) > 0:
        idx = tmp_lst.index(max(tmp_lst))
    return idx


def run_new(video_file, rotate_tag):
    ocr_img_list = []

    config_dict = {
        "ocr_det_config": "./config/det/my_det_r50_db++_td_tr.yml",
        "ocr_rec_config": "./config/rec/my_en_PP-OCRv3_rec.yml"
    }
    my_ocr_process = det_ocr.OCR_process(config_dict)

    # 打开视频文件
    cap = cv2.VideoCapture(video_file)


    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

    if not cap.isOpened():
        print("错误：无法打开视频文件。")
        # exit()
        return None
    count = 0
    results_lst = []
    tmp_res = []
    flag = 0
    save_lst = []
    M = calc_M(rotate_tag)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            count += 1
            if rotate_tag == 'right':
                use_frame = cv2.warpPerspective(frame, M, (2688, 1520))
                use_frame = cv2.rotate(use_frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotate_tag == 'left':
                use_frame = cv2.warpPerspective(use_frame, M, (2688, 1520))
                use_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            ocr_img_list.append(use_frame)
            results = my_ocr_process.process_imgs(ocr_img_list)
            result1, result2, thresh_res = results

            print(result1, result2, thresh_res, flag)
            
            if result1 != ' ' and result2 != ' ':
                tmp_res.append([results, use_frame])
                flag = 0
            elif result1 != ' ':
                tmp_res.append([(result1, '', thresh_res), use_frame])
                flag = 0
            elif result2 != ' ':
                tmp_res.append([('', result2, thresh_res), use_frame])
                flag = 0
            else:
                if(len(tmp_res)!=0 and flag == 8):  #flag 表示在多少帧以后如果一直空则进入判断
                    lt1 = []
                    lt2 = []
                    lt_thresh = []
                    lt_img = []

                    for i in range(len(tmp_res)):
                        word = tmp_res[i][0]
                        w1, w2, s = word
                        if w1 != '':
                            lt1.append(w1)
                        if w2 != '':
                            lt2.append(w2)
                        lt_thresh.append(s)
                        lt_img.append(tmp_res[i][1])

                    save_idx = get_save_idx(lt_thresh)
                    print(save_idx)
                    print(lt_thresh[save_idx])
                    if len(lt_img) > 0:
                        save_lst.append(lt_img[save_idx])
                    f1 = det_ocr.getstr(lt1)
                    f2 = det_ocr.getstr(lt2)
                    results_lst.append((f1, f2))
                    flag = 0
                    tmp_res.clear()            
                else:
                    flag += 1        

            ocr_img_list.clear()

            write_content = f'{result1} {result2}'
            text_size = cv2.getTextSize(write_content, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = width - text_size[0] - 10
            text_y = text_size[1] + 10
            frame_w = cv2.putText(frame, write_content, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            output_video.write(frame_w)

        else:
            break

    # results = my_ocr_process.process_imgs_2(ocr_img_list)
    # print(results)
    # 释放视频流对象和写入对象
    cap.release()

    finalResult = []
    if len(results_lst) > 1:
        for res in results_lst:
            res1, res2 = res
            if res1 != '' and res2 != '':
                finalResult.append(res)

    for i in range(len(save_lst)):
        cv2.imwrite(f'./关键帧存图_{i}.jpg', save_lst[i])

    if len(finalResult) > 2:
        tmpFinal = set(finalResult)
        finalResult = list(tmpFinal)


    return finalResult



def test(imgRoot):
    ocr_img_list = []

    config_dict = {
        "ocr_det_config": "./config/det/my_det_r50_db++_td_tr.yml",
        "ocr_rec_config": "./config/rec/my_en_PP-OCRv3_rec.yml"
        # "ocr_h_rec_config": "./config/rec/my_en_PP-OCRv3_rec.yml",
        # "ocr_v_rec_config": "./config/rec/my_en_PP-OCRv3_rec.yml"
    }
    my_ocr_process = det_ocr.OCR_process(config_dict)

    imgRoot_lst = os.listdir(imgRoot)
    for i in range(len(imgRoot_lst)):
        imgPth = os.path.join(imgRoot, imgRoot_lst[i])
        img = cv2.imread(imgPth)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        ocr_img_list.append(img)

        results = my_ocr_process.process_imgs(ocr_img_list, 'v')
        print(results)
    result_list = []
    result1, result2 = results
    if (result1 != '') & (result2 != ''):
        result_list.append(results)

    print(result_list)

    






# from pympler import asizeof
if __name__ == "__main__":

    root = './1'
    test(root)



    # video = '/home/public210/Public/hjk_workspace/20_lidai/11/202306/video/right/20230629-055756.825-right.mp4'
    # video = '/home/public210/Public/hjk_workspace/20_lidai/11/202306/video/right/20230628-103147.537-right.mp4'
    # if 'left' in video:
    #     rotate_tag = 'left'
    # elif 'right' in video:
    #     rotate_tag = 'right'
    # result = run_new(video, rotate_tag)
    # # print(result)

    # array_str = ', '.join([f'{item[0]}  {item[1]}' for item in result])
    # # 合并路径和数组字符串
    # line_to_write = f"{video}   :{array_str}\n"
    # print(line_to_write)

    # with open('result.txt', 'a') as re:
    #     with open('allpth.txt', 'r') as file:
    #         for line_num, line in enumerate(file, 1):
    #             if line_num >= 604:
    #                 videoPth = line.strip()
    #                 if 'left' in videoPth:
    #                     rotate_tag = 'left'
    #                 elif 'right' in videoPth:
    #                     rotate_tag = 'right'

    #                 result = run_new(videoPth, rotate_tag)
                    
    #                 array_str = ', '.join([f'{item[0]}  {item[1]}' for item in result])

    #                 # 合并路径和数组字符串
    #                 line_to_write = f"{videoPth}   :{array_str}\n"
    #                 re.write(line_to_write)
    #                 print(videoPth, result)
            