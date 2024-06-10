from digit_interface import Digit

from pathlib import Path
import numpy as np

import argparse
import os
import json
import time
import cv2  
from copy import deepcopy



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--outpath",default="output")
    parser.add_argument("--filename",default="testdata")
    parser.add_argument("--runtime",default=3,type=float)
    args = parser.parse_args()

    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)

    d = Digit("D20982")  # Unique serial number
    d.connect()

    # fpsと解像度の設定
    digit_conf = Digit.STREAMS  # ここにconfigが詰まってる
    d.set_resolution(digit_conf["QVGA"])
    fps = digit_conf["QVGA"]["fps"]["30fps"]
    d.set_fps(fps)


    h,w=240,240
    frames=[]
    start_time=time.time()
    is_start=False
    print(f"\033[92mpress 's'\033[0m"+ "to start")
    while time.time()-start_time<args.runtime:
        frame = d.get_frame()
        frame=deepcopy(frame[frame.shape[0]-h:])

        frame_raw=deepcopy(frame) #描画するとグリッド線とかが入っちゃうのでnpyとして保存するのはこれ
        # print(frames.shape) #OK 240x240になってる

        # フレームをリアルタイムで表示
        cv2.imshow('Frame', frame)
        if not is_start and cv2.waitKey(1) & 0xFF == ord('s'):
            is_start=True
            print("collecting data...")

        if is_start: #データ収集開始したら, それ以降はstart_timeを更新しない
            frames.append(frame_raw)
        else:
            start_time=time.time()

    #ウィンドウを閉じる
    cv2.destroyAllWindows()

    frames=np.array(frames)
    np.save(f"{args.outpath}/{args.filename}.npy", frames)
    print(f"\033[92m{args.filename}.npy saved\033[0m")


if __name__ == "__main__":
    main()
