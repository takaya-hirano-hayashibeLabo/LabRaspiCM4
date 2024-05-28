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
import tensorflow as tf
DEVICE=devices()[0]

import time
import os

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
    parser.add_argument("--model_type",default="nn")
    parser.add_argument("--img_size",default=16,type=int)
    parser.add_argument("--savepath",default="result")
    parser.add_argument("--runtime",default=30,type=float,help="推論時間")
    args=parser.parse_args()


    #>> command line args >>
    model_type=args.model_type
    img_size=args.img_size
    runtime=args.runtime
    savepath=Path(__file__).parent/args.savepath/model_type/f"size{img_size}"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        # Save args to savepath

    args_dict = vars(args)
    with open(savepath / 'args.json', 'w') as f:
        json.dump(args_dict, f, indent=4)

    #>> command line args >>


    #>> モデルロード >>
    model_root=Path("/home/neurobo/dev/projects/workspace/prj_2mr_0531/models") #モデル格納ディレクトリ
    modelname="final_model.h5" if model_type=="nn" else "snn.fbz"
    modelpath=model_root/f"{model_type.lower()}/size{img_size}/result"/modelname

    if model_type=="nn".casefold():
        model=tf.keras.models.load_model(modelpath)
    else:
        model=AkidaModel(str(modelpath))
        model.map(DEVICE)
        DEVICE.soc.power_measurement_enabled = True
    model.summary()
    #>> モデルロード >>


    #>> 学習configのロード >>
    conf_path = model_root / f"{model_type.lower()}/size{img_size}/result/conf.json"
    with open(conf_path, 'r') as conf_file:
        conf = json.load(conf_file)
    #>> 学習configのロード >>


    #>> センサの準備 >>
    digit=Digit("D20982")
    digit.connect()
    digit_conf=Digit.STREAMS #ここにconfigが詰まってる
    digit.set_resolution(digit_conf["QVGA"])
    fps=digit_conf["QVGA"]["fps"]["30fps"]
    digit.set_fps(fps)
    #>> センサの準備 >>


    try:
        start_time=time.time()
        while True:

            if time.time()-start_time>runtime: #時間過ぎたらbreak
                print("Runtime exceeded, exiting loop...")
                break


            frame = digit.get_frame()


            #>> クリッピングが必要な場合はクリッピング処理をする >>
            height, width = frame.shape[:2]
            if conf["resize"][0] == conf["resize"][1]:
                min_dim = min(height, width)
                start_x = (width - min_dim) // 2
                start_y = (height - min_dim) // 2
                frame = np.copy(frame[start_y:start_y + min_dim, start_x:start_x + min_dim])
            #>> クリッピングが必要な場合はクリッピング処理をする >>


            filtered_frame = np.copy(frame)
            for filter in conf["filters"]:
                if filter == "roberts":
                    filtered_frame = apply_roberts_filter(np.expand_dims(filtered_frame, axis=0))
                if filter=="canny":
                    filtered_frame = apply_canny_edge_detector(np.expand_dims(filtered_frame, axis=0))
                if filter=="sobel":
                    filtered_frame=apply_sobel_edge_detector(np.expand_dims(filtered_frame, axis=0))

            is_color=filtered_frame.shape[-1]>1
            if is_color:
                input_data = np.expand_dims(cv2.resize(filtered_frame[0], tuple(conf["resize"])), axis=0).astype(np.uint8)
            elif not is_color:
                input_data = np.expand_dims(cv2.resize(filtered_frame[0], tuple(conf["resize"])), axis=0)
                input_data=np.expand_dims(input_data,axis=-1).astype(np.uint8)

            
            #>> modelによる推論 >>
            if model_type=="nn".casefold():
                out=model(input_data/255)

            elif model_type=="snn".casefold():
                input_data = np.repeat(input_data, 512, axis=0)
                out=model.forward(input_data)/255
                out = softmax(np.squeeze(out[0]))
                # out=softmax(np.arange(7))

                power_events = DEVICE.soc.power_meter.events()
                floor_power = DEVICE.soc.power_meter.floor #idle状態の消費電力
                print(f"Timestamp: {power_events[-1].ts}, Voltage: {power_events[-1].voltage} µV, Current: {power_events[-1].current} mA, Power: {power_events[-1].power} mW, Floor: {floor_power}")
                print(model.statistics)
            # print(out)
            predict = np.argmax(out, axis=-1)
            #>> modelによる推論 >>


            #>> 描画 >>
            # フィルタをかけた後のデータをframeの横に描画
            # filtered_view=filtered_frame[0]
            # if not is_color:
            #     filtered_view=cv2.cvtColor(filtered_frame[0], cv2.COLOR_GRAY2RGB)
            
            # # チャンネルごとに分割してグレースケールで描画
            # channels = cv2.split(filtered_view)
            # grayscale_channels = [cv2.cvtColor(ch, cv2.COLOR_GRAY2RGB) for ch in channels]
            # combined_channels = np.hstack(grayscale_channels)
            
            # combined_frame = np.hstack((frame, combined_channels))
            # # 予測結果をフレームに描画
            # cv2.putText(combined_frame, f'Prediction: {predict}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # # ヒストグラムを描画してフレームの横に追加
            # combined_frame = draw_histogram(out, combined_frame)
            # cv2.imshow('Frame', combined_frame) # フレームを表示
            # #>> 描画 >>

            
            # if cv2.waitKey(10) & 0xFF == ord('q'): # qキーが押されたかチェック
            #     print("Exiting loop...")
            #     break
            
            time.sleep(1.0 / fps)
    finally:
        digit.disconnect()
        cv2.destroyAllWindows() # 追加


if __name__ == "__main__":
    main()
