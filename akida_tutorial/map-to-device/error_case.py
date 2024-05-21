"""
modelのIPバージョンが2だとメモリに乗らない
"""

from akida import Model as AkidaModel
from akida import devices,Device
DEVICE=devices()[0]

def main():

    modelpath="/home/neurobo/dev/projects/touch-recognition/coins/case1/snn.fbz" #version2のモデル(間違えて消しちゃった)
    model=AkidaModel(str(modelpath))
    print(f"model ip version : {model.ip_version}")
    model.map(DEVICE,hw_only=True)
    model.summary()

if __name__=="__main__":
    main()
