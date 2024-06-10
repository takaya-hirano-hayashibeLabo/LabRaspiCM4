import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import tensorflow as tf
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from digit_interface import Digit
import time
from copy import deepcopy   


from utils import load_json_to_dict,img2height


def plot_frames(frame_prev, frame_new):
    """
    Plot the previous and new frames side by side.
    
    Parameters:
    frame_prev (np.ndarray): Previous frame.
    frame_new (np.ndarray): New frame.
    """
    plt.figure(figsize=(10, 5))
    
    # Convert to grayscale if the frames have 3 channels
    if len(frame_prev.shape) == 3:
        frame_prev = cv2.cvtColor(frame_prev, cv2.COLOR_RGB2GRAY)
    if len(frame_new.shape) == 3:
        frame_new = cv2.cvtColor(frame_new, cv2.COLOR_RGB2GRAY)
    
    # Plot previous frame
    plt.subplot(1, 2, 1)
    plt.title("Previous Frame")
    plt.imshow(frame_prev, cmap="gray")
    plt.axis('off')

    # Plot new frame
    plt.subplot(1, 2, 2)
    plt.title("New Frame")
    plt.imshow(frame_new, cmap="gray")
    plt.axis('off')

    plt.show()

def calculate_phase_correlation(frame_prev, frame_new):
    """
    Calculate the motion vector between two frames using Phase Correlation method.
    
    Parameters:
    frame_prev (np.ndarray): Previous frame.
    frame_new (np.ndarray): New frame.
    
    Returns:
    tuple: (dx, dy) motion vector.
    """
    # Convert frames to grayscale if they are not already
    if len(frame_prev.shape) == 3:
        frame_prev_gray = cv2.cvtColor(frame_prev, cv2.COLOR_RGB2GRAY)
    else:
        frame_prev_gray = frame_prev

    if len(frame_new.shape) == 3:
        frame_new_gray = cv2.cvtColor(frame_new, cv2.COLOR_RGB2GRAY)
    else:
        frame_new_gray = frame_new


    # Perform phase correlation
    shift,peak = cv2.phaseCorrelate(np.float32(frame_prev_gray), np.float32(frame_new_gray))
    print(f"shift : {shift}, peak : {peak}")

    # plot_frames(frame_prev,frame_new)
    
    return shift


def update_map(height_map,vector_shift,vector_origin,height_new):
    """
    :param height_map : 
    :param vector_shift : 
    :param vector_origin : mapの原点から見た新しいフレームの原点
    """

    #>> mapサイズの更新 >>
    vector_origin_new=vector_origin+vector_shift
    vx,vy=vector_origin_new

    h_map,w_map=height_map.shape
    h_height,w_height=height_new.shape

    pad_w=(0,0)
    if vx>=0 and w_height-(w_map-vx)>0: #右側にパディングするとき
        pad_w=(0,w_height-(w_map-vx))
    elif vx<0: #左にパディングするとき
        pad_w=(abs(vx),0)

    pad_h=(0,0)
    if vy>=0 and h_height-(h_map-vy)>0: #下にパディング
        pad_h=(0,h_height-(h_map-vy))
    elif vy<0: #上にパディング
        pad_h=(abs(vy),0)

    height_map_new=np.pad(
        height_map,
        (pad_h,pad_w), mode="edge" #端と同じ値でパディング
    )

    is_padl,is_padu=vx<0,vy<0 #左、上にパディングしたか
    vector_origin_new=vector_origin_new-np.array([vx*is_padl,vy*is_padu]) #左、上にパディングした場合は原点更新
    #<< mapサイズの更新 <<


    #>> 新しいフレームの高さマップを代入 >>
    print("height map shape:",height_map_new.shape)
    print("height new shape: ",height_new.shape)
    x_start, y_start = vector_origin_new
    x_end, y_end = x_start + w_height, y_start + h_height
    print(f"ys:{y_start}, ye:{y_end}/ xs:{x_start}, xe:{x_end}")
    height_map_new[y_start:y_end,x_start:x_end] = (height_map_new[y_start:y_end,x_start:x_end]+height_new)/2 #平均で更新. ここの更新アルゴリズムをもっといい感じにできそう
    #<< 新しいフレームの高さマップを代入 <<


    # ##>> 境界をなじませる >>
    # band=15 #px
    # # 上側の境界を線形補間
    # s1=y_start-int(band/2) if y_start-int(band/2)>0 else 0
    # s2=y_start+int(band/2)
    # alpha=((height_map_new[s2,x_start:x_end]-height_map_new[s1,x_start:x_end])/(s2-s1)).reshape(1,-1)
    # indices=np.repeat(np.arange(s2-s1).reshape(-1,1),w_height,axis=1)
    # # print("s2,s1,a,i",s2,s1,alpha.shape,indices.shape)
    # height_map_new[s1:s2,x_start:x_end]=height_map_new[s1,x_start:x_end]+alpha*indices

    # #下側の境界を線形補完
    # e1=y_end-int(band/2)
    # e2=y_end+int(band/2) if y_end+int(band/2)<height_map_new.shape[0] else height_map_new.shape[0]-1
    # alpha=((height_map_new[e2,x_start:x_end]-height_map_new[e1,x_start:x_end])/(e2-e1)).reshape(1,-1)
    # indices=np.repeat(np.arange(e2-e1).reshape(-1,1),w_height,axis=1)
    # height_map_new[e1:e2,x_start:x_end]=height_map_new[e1,x_start:x_end]+alpha*indices


    # #左側の境界を線形補完
    # s1=x_start-int(band/2) if x_start-int(band/2)>0 else 0
    # s2=x_start+int(band/2)
    # alpha=((height_map_new[y_start:y_end,s2]-height_map_new[y_start:y_end,s1])/(s2-s1)).reshape(-1,1)
    # indices=np.repeat(np.arange(s2-s1).reshape(1,-1),h_height,axis=0)
    # height_map_new[y_start:y_end,s1:s2]=height_map_new[y_start:y_end,s1].reshape(-1,1)+alpha*indices


    # #右側の境界を線形補完
    # e1=x_end-int(band/2)
    # e2=x_end+int(band/2) if x_end+int(band/2)<height_map_new.shape[1] else height_map_new.shape[1]-1
    # alpha=((height_map_new[y_start:y_end,e2]-height_map_new[y_start:y_end,e1])/(e2-e1)).reshape(-1,1)
    # indices=np.repeat(np.arange(e2-e1).reshape(1,-1),h_height,axis=0)
    # height_map_new[y_start:y_end,e1:e2]=height_map_new[y_start:y_end,e1].reshape(-1,1)+alpha*indices



    return height_map_new,vector_origin_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelpath",required=True)
    parser.add_argument("--freq",default=2,type=int, help="マップを更新する周波数")
    parser.add_argument("--runtime",default=10.0,type=float, help="実行時間")
    args = parser.parse_args()


    savepath = Path(__file__).parent / "result"
    if not os.path.exists(savepath):
        os.makedirs(savepath)


    print("loading json...")
    data_conf=load_json_to_dict(Path(args.modelpath)/"args.json")["train_data_conf"] #学習データのconf. pxサイズが入ってる
    print("\033[92mdone\033[0m\n")  # Green text 


    # モデルの生成
    modelpath=Path(args.modelpath)/"final_model.h5"
    model = tf.keras.models.load_model(modelpath)
    # model.summary()


    # >> 背景データの読み込み >>
    backgroundpath=Path(__file__).parent/"background"/"background.npy"
    background=np.load(backgroundpath)[30] #なにもないときのデータ
    # << 背景データの読み込み <<


    #>> Digitセンサの設定 >>
    print("setting up digit...")
    d = Digit("D20982")  # Unique serial number
    d.connect()

    # fpsと解像度の設定
    digit_conf = Digit.STREAMS  # ここにconfigが詰まってる
    d.set_resolution(digit_conf["QVGA"])
    fps = digit_conf["QVGA"]["fps"]["30fps"]
    d.set_fps(fps)
    start_time=time.time()
    while time.time()-start_time<10: #センサが安定するまで少し待つ
        tmp=d.get_frame()
        tmp=deepcopy(tmp[int(tmp.shape[0]-data_conf["px_size"][0]):])
        cv2.imshow("digit frame",tmp)
        time.sleep(1/fps)
    cv2.destroyAllWindows()
    print("\033[92mdone\033[0m\n")  # Green text 
    #<< Digitセンサの設定 <<


    #>> マップの初期化 >>
    height_bg=img2height(background,model,data_conf["px_size"])

    frame_new=d.get_frame() #まずは1フレーム分とる
    frame_new=deepcopy(frame_new[int(frame_new.shape[0]-data_conf["px_size"][0]):]) #上は捨てる
    height_new=img2height(frame_new,model,data_conf["px_size"])-height_bg
    height_map=np.copy(height_new)
    height_prev=np.copy(height_new)
    vector_origin=np.array([0,0]) #原点から見た新しいフレームの原点
    # plot_frames(frame_new,height/np.max(height))
    #>> マップの初期化 >>



    interval = 1.0 / args.freq  # Calculate the interval in seconds
    start_runtime = time.time()  # Record the start time of the runtime
    while True:
        print(f"elapsed time: {time.time()-start_runtime}")
        start_time = time.time()  # Record the start time of the loop

        frame_new = d.get_frame()
        frame_new=deepcopy(frame_new[int(frame_new.shape[0]-data_conf["px_size"][0]):]) #上は捨てる
        print(frame_new.shape)
        height_new = img2height(frame_new, model, data_conf["px_size"]) - height_bg

        mask_prev = height_prev / np.max(height_prev)
        mask_prev[mask_prev < 0.5] = 0
        mask_new = height_new / np.max(height_new)
        mask_new[mask_new < 0.5] = 0
        vector_shift = calculate_phase_correlation(mask_prev, mask_new)
        vector_shift = np.array([-round(vector_shift[0]), -round(vector_shift[1])])  # 整数化&反転
        print(f"frame : {vector_shift}")

        height_map, vector_origin = update_map(height_map, vector_shift, vector_origin, height_new)

        print("map shape: ", height_map.shape)
        print("orign vector: ", vector_origin)

        height_prev = np.copy(height_new)

        # Display the height map in real-time
        cv2.imshow('Height Map', height_map / np.max(height_map))  # Normalize for display
        cv2.imshow('Frame New', frame_new)  # Display the new frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        elapsed_time = time.time() - start_time  # Calculate the elapsed time
        sleep_time = max(0, interval - elapsed_time)  # Calculate the remaining time to sleep
        time.sleep(sleep_time)  # Pause the loop for the remaining interval duration

        # Check if runtime has exceeded
        if time.time() - start_runtime > args.runtime:
            break

    d.disconnect()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()





