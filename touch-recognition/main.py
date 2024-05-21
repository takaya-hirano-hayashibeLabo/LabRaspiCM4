import argparse
from pathlib import Path
import numpy as np
import cv2  
# import tensorflow as tf
import json
from digit_interface import Digit
import akida
from akida import Model as AkidaModel
from akida import devices,Device
DEVICE=devices()[0]

import time

from filters import *



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

    while True:

        frame = digit.get_frame()


        filtered_frame=np.copy(frame)
        for filter in conf["filters"]:
            if filter=="roberts":
                filtered_frame=apply_roberts_filter(np.expand_dims(filtered_frame,axis=0))

        input_data=np.expand_dims(cv2.resize(filtered_frame[0],(72,96)),axis=0).astype(np.uint8)

        # predict = model.predict_classes(input_data)
        out=np.squeeze(model.forward(input_data))
        predict=np.argmax(out,axis=-1)

        # フィルタをかけた後のデータをframeの横に描画
        combined_frame = np.hstack((frame, filtered_frame[0]))

        # 予測結果をフレームに描画
        cv2.putText(combined_frame, f'Prediction: {predict}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Frame', combined_frame)  # フレームを表示

        if cv2.waitKey(1) & 0xFF == ord('q'):  # qキーが押されたかチェック
            print("Exiting loop...")
            break

        time.sleep(1.0/fps)
            
    digit.disconnect()
    cv2.destroyAllWindows()  # 追加

if __name__=="__main__":
    main()
