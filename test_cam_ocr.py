#!/usr/bin/python3
# -*- coding:utf-8 -*-
import sys
from paddleocr import PaddleOCR#
sys.path.append('/workspace/hugo_py')

import cv2
import time
import Speech
from camera import JetCamera
import traceback
import queue
import threading

import difflib

cap_w = 640
cap_h = 360
cap_fps = 5

def string_similar(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()

def samelike(result0, result1):
        s1 = ''
        s2 = ''
        for i in result0:
            s1 += result0[0][1][0]

        for i in result1:
            s2 += result1[0][1][0]

        return string_similar(s1, s2)



def main():
    # start camera
    cam = JetCamera(cap_w, cap_h, cap_fps)
    cam.open()
    b_exit = False

    # start speech
    Speech.SetReader(Speech.Reader_Type["Reader_XuXiaoBao"])
    Speech.SetVolume(1)
    Speech.SetSpeed(5)
    speech_que = queue.Queue(maxsize=3)

    def speech_run():
        last_result=[]
        while not b_exit:
            try:
                result = speech_que.get(block=True, timeout=0.1)
            except:
                continue

            print('get text .... %s'  % result)

            try:
                if last_result:
                    likehood = samelike(last_result, result)

                    if likehood >= 0.6 and len(last_result) == len(result):
                        Speech.Block_Speech_text("请翻页")
                        last_result = result 
                        continue 
                last_result = result 

            except:
                traceback.print_exc()
                pass 

            text_lst = []
            for line in result:
                text = line[1][0]
                text_lst.append(text)

            for text in text_lst:
                Speech.Block_Speech_text(text)

    t_speech = threading.Thread(target=speech_run)
    t_speech.start()

    # start ocr
    ocr_queue = queue.Queue(maxsize=1)
    def ocr_run():
        global ocr_init
        ocr = PaddleOCR(use_angle_cls=True, use_gpu=True, use_tensorrt=False, gpu_mem=3000, det_max_side_len=960, rec_batch_num=2)
        cnt = 0
        last_result = []
        while not b_exit:
            try:
                frame = ocr_queue.get(block=True, timeout=0.1)
            except:
                continue 
            
            print('ocr got one frame ....')
            t0 = time.time()
            result = ocr.ocr(frame, cls=True)
            t1 = time.time()

            #try:
            #    speech_que.put_nowait(result)
            #except:
            #    pass
            if len(result) < 1:
                continue 

            try:
                if last_result:
                    likehood = samelike(last_result, result)

                    if likehood >= 0.6: # and len(last_result) == len(result):
                        Speech.Block_Speech_text("请翻页")
                        last_result = result
                        continue

                last_result = result

            except:
                traceback.print_exc()
                pass

            if 1:
                text_lst = []
                for line in result:
                    text = line[1][0]
                    text_lst.append(text)
                    print(line)

                for text in text_lst:
                    Speech.Block_Speech_text(text)

            cnt += 1

            if cnt % 1 == 0:
                print("frame cnt [%d] ocr detect delay = %.1fms" % (cnt, (t1 - t0) * 1000))

    t_ocr = threading.Thread(target=ocr_run)
    t_ocr.start()

    # do live-camera-detect
    out_win = "ocr"
    cv2.namedWindow(out_win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(out_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        try:
            ret, frame = cam.read()
            if not ret:
                print('get frame failed ')

                break
           
            if 1:
                try:
                    ocr_queue.put_nowait(frame)
                except:
                    pass
            # mirror 
            frame1 = frame[:,::-1]
            cv2.imshow(out_win, frame1)
            cv2.waitKey(1)
        except:
            traceback.print_exc()
            break

    print('... done ')
    b_exit = True
    cam.close()


if __name__ == '__main__':
    main()
