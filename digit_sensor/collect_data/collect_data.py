from digit_interface import Digit

from pathlib import Path
import numpy as np

import argparse
from pathlib import Path
import os
import json
import time

import threading

# ユーザーの入力を監視するフラグ
exit_flag = False

def monitor_input():
    global exit_flag
    while True:
        user_input = input()
        if user_input == 'q':
            exit_flag = True
            break

def collect_data(digit_sensor:Digit,fps):
    global exit_flag
    frames=[]

    while not exit_flag:
        frames+=[digit_sensor.get_frame()]
        time.sleep(1.0/fps)
    digit_sensor.disconnect()

    return frames

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--label",required=True)
    parser.add_argument("--fps",default=30,type=int)
    parser.add_argument("--file_name",required=True)
    parser.add_argument("--relative_parentpath", required=True)

    args=parser.parse_args()


    # 入力監視用のスレッドを開始
    input_thread = threading.Thread(target=monitor_input)
    input_thread.start()


    d = Digit("D20982") # Unique serial number
    d.connect()


    #>> fpsと解像度の設定 >>
    digit_conf=Digit.STREAMS #ここにconfigが詰まってる
    d.set_resolution(digit_conf["QVGA"])
    fps=digit_conf["QVGA"]["fps"]["30fps"]
    d.set_fps(fps)
    #>> fpsと解像度の設定 >>


    print("\ncollecting digit data...")
    print("***put 'q' key to quit data collection***")
    frames=collect_data(d,fps)


    input_thread.join()


    #>> データの保存 >>
    data_path=Path(__file__).parent/args.relative_parentpath/args.label
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    filename=args.file_name if ".npy" in args.file_name else args.file_name+".npy"
    np.save(str(data_path/filename),np.array(frames))


    # JSONファイルに保存
    with open(data_path/f'{args.file_name}.json', 'w') as json_file:
        json.dump(vars(args), json_file, indent=3)
    #>> データの保存 >>



if __name__=="__main__":
    main()