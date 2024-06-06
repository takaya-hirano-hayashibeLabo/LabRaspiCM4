#最後に動画を保存する用のコード

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", required=True)
    args = parser.parse_args()

    modelpath = Path(args.target_dir) / "snn.fbz"
    model = AkidaModel(str(modelpath))
    print(f"model ip version : {model.ip_version}")
    model.map(DEVICE, hw_only=True)
    model.summary()

    conf_path = Path(args.target_dir) / "conf.json"
    with open(conf_path, 'r') as conf_file:
        conf = json.load(conf_file)

    digit = Digit("D20982")
    digit.connect()
    digit_conf = Digit.STREAMS
    digit.set_resolution(digit_conf["QVGA"])
    fps = digit_conf["QVGA"]["fps"]["30fps"]
    digit.set_fps(fps)

    # VideoWriterの設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # コーデック
    out_video = None  # VideoWriterを後で初期化するためにNoneに設定

    start_time = time.time()  # Start the timer

    try:
        while True:
            frame = digit.get_frame()
            filtered_frame = np.copy(frame)
            for filter in conf["filters"]:
                if filter == "roberts":
                    filtered_frame = apply_roberts_filter(np.expand_dims(filtered_frame, axis=0))
                if filter == "canny":
                    filtered_frame = apply_canny_edge_detector(np.expand_dims(filtered_frame, axis=0))
                if filter == "sobel":
                    filtered_frame = apply_sobel_edge_detector(np.expand_dims(filtered_frame, axis=0))
            is_color = filtered_frame.shape[-1] > 1
            if is_color:
                input_data = np.expand_dims(cv2.resize(filtered_frame[0], tuple(conf["resize"])), axis=0).astype(np.uint8)
            elif not is_color:
                input_data = np.expand_dims(cv2.resize(filtered_frame[0], tuple(conf["resize"])), axis=0)
                input_data = np.expand_dims(input_data, axis=-1).astype(np.uint8)
            out = softmax(np.squeeze(model.forward(input_data)) / 255)
            predict = np.argmax(out, axis=-1)

            filtered_view = filtered_frame[0]
            if not is_color:
                filtered_view = cv2.cvtColor(filtered_frame[0], cv2.COLOR_GRAY2RGB)

            channels = cv2.split(filtered_view)
            grayscale_channels = [cv2.cvtColor(ch, cv2.COLOR_GRAY2RGB) for ch in channels]
            combined_channels = np.hstack(grayscale_channels)

            combined_frame = np.hstack((frame, combined_channels))
            cv2.putText(combined_frame, f'Prediction: {predict}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            combined_frame = draw_histogram(out, combined_frame)

            # フレームサイズを確認
            # print(f"Combined frame size: {combined_frame.shape}")

            # VideoWriterの初期化（最初のフレームでのみ行う）
            if out_video is None:
                frame_height, frame_width = combined_frame.shape[:2]
                out_video = cv2.VideoWriter(str(Path(args.target_dir)/'output.mp4'), fourcc, fps, (frame_width, frame_height))

            # フレームを動画に書き込む
            out_video.write(combined_frame.astype(np.uint8))
            # フレームを表示する
            cv2.imshow('Combined Frame', combined_frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                print("Exiting loop...")
                break

            # Check if 30 seconds have passed
            if time.time() - start_time > 30:
                print("30 seconds elapsed. Exiting loop...")
                break

            time.sleep(1.0 / fps)
    finally:
        digit.disconnect()
        cv2.destroyAllWindows()
        if out_video is not None:
            out_video.release()  # 動画ファイルを閉じる

if __name__ == "__main__":
    main()
