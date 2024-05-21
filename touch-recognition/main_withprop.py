import argparse
from pathlib import Path
import numpy as np
import cv2  
import matplotlib.pyplot as plt
import json
from digit_interface import Digit
import akida
from akida import Model as AkidaModel
from akida import devices,Device
DEVICE=devices()[0]

import time

from filters import *

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def draw_histogram(out, frame):
    hist_height = 300
    hist_width = frame.shape[1]
    bin_height = int(hist_height / len(out))

    hist_img = np.zeros((hist_height + 50, hist_width, 3), dtype=np.uint8)  # Extra space for x-axis labels

    # Define a colormap
    colormap = plt.get_cmap('viridis')
    colors = (colormap(np.linspace(0, 1, len(out)))[:, :3] * 255).astype(np.uint8)

    for i in range(len(out)):
        bin_val = int(out[i] * hist_width)
        color = tuple(map(int, colors[i]))
        cv2.rectangle(hist_img, (0, i * bin_height), (bin_val, (i + 1) * bin_height), color, -1)
        cv2.putText(hist_img, str(i), (5, (i + 1) * bin_height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Add x-axis labels
    for i in range(11):
        x_pos = int(i * hist_width / 10)
        cv2.putText(hist_img, f'{i / 10:.1f}', (x_pos, hist_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Combine the frame and histogram image top and bottom
    combined_frame = np.vstack((frame, hist_img))

    return combined_frame


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--target_dir",required=True)
    args=parser.parse_args()

    # print(f"PCIe device : {DEVICE.desc}, {DEVICE.version}, {akida.NSoC_v2}")

    modelpath=Path(args.target_dir)/"snn.fbz"
    model=AkidaModel(str(modelpath))
    print(f"model ip version : {model.ip_version}")
    model.map(DEVICE,hw_only=True)
    model.summary()

    conf_path = Path(args.target_dir) / "conf.json"
    with open(conf_path, 'r') as conf_file:
        conf = json.load(conf_file)

    digit=Digit("D20982")
    digit.connect()
    digit_conf=Digit.STREAMS #ここにconfigが詰まってる
    digit.set_resolution(digit_conf["QVGA"])
    fps=digit_conf["QVGA"]["fps"]["30fps"]
    digit.set_fps(fps)

    try:
        while True:
            frame = digit.get_frame()
            filtered_frame = np.copy(frame)
            for filter in conf["filters"]:
                if filter == "roberts":
                    filtered_frame = apply_roberts_filter(np.expand_dims(filtered_frame, axis=0))
                if filter=="canny":
                    filtered_frame = apply_canny_edge_detector(np.expand_dims(filtered_frame, axis=0))

            is_color=filtered_frame.shape[-1]>1
            if is_color:
                input_data = np.expand_dims(cv2.resize(filtered_frame[0], (72, 96)), axis=0).astype(np.uint8)
            elif not is_color:
                input_data = np.expand_dims(cv2.resize(filtered_frame[0], (72, 96)), axis=0)
                input_data=np.expand_dims(input_data,axis=-1).astype(np.uint8)
            out = softmax(np.squeeze(model.forward(input_data))/255)
            predict = np.argmax(out, axis=-1)

            # フィルタをかけた後のデータをframeの横に描画
            filtered_view=filtered_frame[0]
            if not is_color:
                filtered_view=cv2.cvtColor(filtered_frame[0], cv2.COLOR_GRAY2RGB)
            combined_frame = np.hstack((frame, filtered_view))

            # 予測結果をフレームに描画
            cv2.putText(combined_frame, f'Prediction: {predict}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # ヒストグラムを描画してフレームの横に追加
            combined_frame = draw_histogram(out, combined_frame)
            cv2.imshow('Frame', combined_frame) # フレームを表示

            if cv2.waitKey(10) & 0xFF == ord('q'): # qキーが押されたかチェック
                print("Exiting loop...")
                break
            
            time.sleep(1.0 / fps)
    finally:
        digit.disconnect()
        cv2.destroyAllWindows() # 追加

if __name__ == "__main__":
    main()
