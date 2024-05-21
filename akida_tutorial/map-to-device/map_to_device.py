from akida import Model as AkidaModel
from akida import devices,Device
DEVICE=devices()[0]

def main():

    modelpath="v1model.fbz"
    model=AkidaModel(str(modelpath))
    print(f"model ip version : {model.ip_version}")
    model.map(DEVICE,hw_only=True)
    model.summary()

if __name__=="__main__":
    main()
