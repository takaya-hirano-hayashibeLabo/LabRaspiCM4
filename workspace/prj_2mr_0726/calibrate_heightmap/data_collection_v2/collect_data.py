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
    parser.add_argument("--ball_radius", default=2.5, type=float)
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


    grid_idx=0
    pr=34 # 描画する円の半径 [px]
    h,w=240,240
    for row in range(3):
        for col in range(3):

            frames=[]

            px_center=40+80*col # 描画する円の中心のx座標 [px]
            py_center=40+80*row # 描画する円の中心のy座標 [px]
            print(f"grid_idx:{grid_idx}, px:{py_center}, py:{px_center}")
            print(f"\033[92mpress 's'\033[0m"+ "to start")

            start_time=time.time()
            is_start=False
            while time.time()-start_time<args.runtime:
                frame = d.get_frame()
                frame=deepcopy(frame[frame.shape[0]-h:])

                frame_raw=deepcopy(frame) #描画するとグリッド線とかが入っちゃうのでnpyとして保存するのはこれ
                # print(frames.shape) #OK 240x240になってる

                # # フレームに円を描画
                cv2.circle(frame, (px_center, py_center), pr, (0, 255, 0), 1)
                # フレームに中心点を描画
                cv2.circle(frame, (px_center, py_center), 3, (0, 0, 255), -1)  # 小さな塗りつぶし円

                # フレームにグリッド線を描画
                for i in range(1, 3):
                    # 垂直線
                    cv2.line(frame, (80 * i, 0), (80 * i, h), (255, 0, 0), 1)
                    # 水平線
                    cv2.line(frame, (0, 80 * i), (w, 80 * i), (255, 0, 0), 1)

                # フレームをリアルタイムで表示
                cv2.imshow('Frame', frame)
                if not is_start and cv2.waitKey(1) & 0xFF == ord('s'):
                    is_start=True
                    print("collecting data...")

                if is_start: #データ収集開始したら, それ以降はstart_timeを更新しない
                    frames.append(frame_raw)
                else:
                    start_time=time.time()

            print(f"\033[92mgrid [{grid_idx}] done\033[0m\n")
            grid_idx+=1

            frames=np.array(frames)
            np.save(f"{args.outpath}/grid{grid_idx}.npy", frames)

            grid_data={
                "px_center":px_center,
                "py_center":py_center,
                "radius_pix":pr,
                "radius_mm":args.ball_radius,
                "fps":30
            }
            with open(f"{args.outpath}/grid{grid_idx}.json", "w") as f:
                json.dump(grid_data, f,indent=4)

        #ウィンドウを閉じる
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
