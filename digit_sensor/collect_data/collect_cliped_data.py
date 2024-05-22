from digit_interface import Digit

from pathlib import Path
import numpy as np

import argparse
from pathlib import Path
import os
import json
import time
import cv2  

def collect_data(digit_sensor:Digit, fps):
    frames = []

    while True:
        frame = digit_sensor.get_frame()
        
        # 正方形にクリッピングする処理を追加
        height, width = frame.shape[:2]
        if height != width:
            min_dim = min(height, width)
            start_x = (width - min_dim) // 2
            start_y = (height - min_dim) // 2
            frame = frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

        frames.append(frame)
        cv2.imshow("frame", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        time.sleep(1.0 / fps)
    
    digit_sensor.disconnect()
    cv2.destroyAllWindows()

    return frames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--file_name", required=True)
    parser.add_argument("--relative_parentpath", required=True)

    args = parser.parse_args()

    d = Digit("D20982")  # Unique serial number
    d.connect()

    # fpsと解像度の設定
    digit_conf = Digit.STREAMS  # ここにconfigが詰まってる
    d.set_resolution(digit_conf["QVGA"])
    fps = digit_conf["QVGA"]["fps"]["30fps"]
    d.set_fps(fps)

    print("\ncollecting digit data...")
    print("***put 'q' key to quit data collection***")
    frames = collect_data(d, fps)

    # データの保存
    data_path = Path(__file__).parent / args.relative_parentpath / args.label
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    filename = args.file_name if ".npy" in args.file_name else args.file_name + ".npy"
    np.save(str(data_path / filename), np.array(frames))

    # JSONファイルに保存
    with open(data_path / f'{args.file_name}.json', 'w') as json_file:
        json.dump(vars(args), json_file, indent=3)

if __name__ == "__main__":
    main()