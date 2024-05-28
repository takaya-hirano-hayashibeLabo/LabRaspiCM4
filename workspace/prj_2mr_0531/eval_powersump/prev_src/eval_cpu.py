import argparse
from pathlib import Path
import numpy as np
import cv2  
import matplotlib.pyplot as plt
import json
import tensorflow as tf

import time
import os

from filters import *



def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--img_size",default=16,type=int)
    parser.add_argument("--savepath",default="result")
    parser.add_argument("--is_runmodel", action='store_true')
    args=parser.parse_args()


    #>> command line args >>
    model_type="nn"
    img_size=args.img_size
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
    modelname="final_model.h5"
    modelpath=model_root/f"{model_type.lower()}/size{img_size}/result"/modelname

    model=tf.keras.models.load_model(modelpath)
    model.summary()
    #>> モデルロード >>


    #>> テストデータのロード >>
    test_root=Path("/home/neurobo/dev/projects/workspace/prj_2mr_0531/test_data")
    test_in=np.load(test_root/f"size{img_size}/test_data.npy")
    test_label=np.load(test_root/f"size{img_size}/test_labels.npy")
    #>> テストデータのロード >>


    batchsize=512
    num_batches = len(test_in) // batchsize

    start_time=time.time()
    acc=0
    datasize=0
    for i in range(num_batches):
        
        batch_inputs = test_in[i * batchsize:(i + 1) * batchsize]
        batch_labels = test_label[i * batchsize:(i + 1) * batchsize].astype(np.int8)
        
        if args.is_runmodel:
            out = np.squeeze(model(batch_inputs/255))
        else:
            out=np.arange(7)
        predict = np.argmax(out, axis=-1)
        acc+=np.sum(predict==batch_labels)
        datasize+=len(batch_inputs)
        #>> modelによる推論 >>

    acc=acc/datasize
    print(f"\n{'='*40}\nAccuracy: {acc:.2%}\n{'='*40}")       

    runtime=time.time()-start_time     
    print(f"Total Runtime: {runtime:.2f} seconds\n{'='*40}")

if __name__ == "__main__":
    main()
