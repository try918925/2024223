



class ImageCaptureProcess(Process):
    def __init__(self, camera_info, queue):
        super().__init__()
        self.camera_id = camera_info['id']
        self.file_path = camera_info['file_path']
        self.direction = camera_info['direction']
        self.roi = camera_info.get('roi', None)
        self.queue = queue
        self.front_rear = ("front", "rear")
        self.left_right_top = ("right", "left", "top")
        self.dsize = (1066, 600)
        self.roi_cols = (self.roi[2], self.roi[3])
        self.roi_rows = (self.roi[0], self.roi[1])
        self.counter = 0

    def run(self):
        try:
            decoder = cv2.cudacodec.createVideoReader(self.file_path)
            while True:
                ret, frame = decoder.nextFrame()
                if not ret or frame is None:
                    decoder = cv2.cudacodec.createVideoReader(self.file_path)
                    print(f"{self.direction}:识别断流........")
                    continue
                if self.direction in self.front_rear:
                    self.counter += 1
                    if self.counter % 2 == 0:
                        frame_crop = frame.colRange(self.roi_cols[0], self.roi_cols[1]).rowRange(self.roi_rows[0],
                                                                                                 self.roi_rows[1])
                        frame_cpu = frame_crop.download()  # 裁剪后的图像下载到 CPU
                        frame_cpu = frame_cpu[:, :, :3]  # 裁剪通道
                        self.queue.put((self.camera_id, frame_cpu))
                        self.counter = 0
                else:
                    frame_cpu = self.crop_images(frame,self.direction)
                    self.queue.put((self.camera_id, frame_cpu))
        except Exception as error:
            print(f"ImageCaptureProcess---{self.direction}:图片读取有问题:{error}")


    def crop_images(self,img, v):
        if v == 'right':
            trapezoid = np.array([[450, 637], [2120, 684], [456, 880], [2118, 811]], dtype=np.float32)
            rectangle = np.array([[200, 0], [1040, 0], [200, 300], [1040, 300]], dtype=np.float32)
            # 计算透视变换矩阵
            M = cv2.getPerspectiveTransform(trapezoid, rectangle)
            # 进行透视变换
            result = cv2.warpPerspective(img, M, (2560, 1440))
            result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)
            result = result[130:1220, 1140:1440]


        elif v == 'left':
            trapezoid = np.array([[392, 635], [2130, 560], [393, 795], [2126, 872]], dtype=np.float32)
            rectangle = np.array([[200, 0], [1040, 0], [200, 300], [1040, 300]], dtype=np.float32)
            # 计算透视变换矩阵
            M = cv2.getPerspectiveTransform(trapezoid, rectangle)
            # 进行透视变换
            result = cv2.warpPerspective(img, M, (2560, 1440))
            result = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
            result = result[1450:2540, 0:300]

        elif v == 'top':
            trapezoid = np.array([[224, 625], [2377, 604], [224, 890], [2377, 883]], dtype=np.float32)  # 左上、右上、左下、右下
            rectangle = np.array([[0, 0], [840, 0], [0, 200], [840, 200]], dtype=np.float32)
            # 计算透视变换矩阵
            M = cv2.getPerspectiveTransform(trapezoid, rectangle)
            # 进行透视变换
            result = cv2.warpPerspective(img, M, (840, 200))
            result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)
        return result