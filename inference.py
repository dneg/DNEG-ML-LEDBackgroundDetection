import numpy as np
import argparse, os, torch, cv2
from collections import OrderedDict

from src.Networks.UnetV1.UnetV1_config import UnetV1Config
from src.Networks.UnetV1.UnetV1_component import UnetV1

# takes the result of a batch (B,H,W) image ()
def postprocess(x, original):

    H, W,_ = original.shape #width and height of output

    x = x.squeeze() #get rid of 1-channel dimension
    B,TH,TW = x.shape  #batch (num tiles), tile height, tile width

    Hp = H if H%TH == 0 else H+(TH-H%TH)  #padded image dimensions
    Wp = W if W%TW == 0 else W+(TW-W%TW)

    assert(Wp*Hp == B*TH*TW)

    x = x.view(int(Hp/TH), int(Wp/TW), TH, TW)  #split B dims into TH and TW: (TH,TW,H,W)
    x = x.permute((0,2,1,3)) #(TH,H,TW,W)
    x = x.contiguous().view(Hp,Wp)
    x = x[0:H,0:W]  #crop out the padding

    #turn into a 3 channel image of ints 0..255
    x = x.numpy() * 255
    x = np.dstack((x,x,x)).astype('uint8')
    return x

#takes a  cv2 image (H,W,C) with channels = (BGR)
#returns a batch of tensors to be (B,C,H,W) with channels = (RGB)
def preprocess(x):
    T = 128 #tile size
    H, W, C = x.shape
    Hpad = 0 if H%T==0 else T-H%T
    Wpad = 0 if W%T==0 else T-W%T
    x = np.pad(x, (((0, Hpad),(0, Wpad), (0,0))), 'constant')

    #Frame from cv2 is (H,W,C) with channels = (BGR)
    #we want a batch of tensors to be (B,C,H,W) with channels = (RGB)
    x = cv2.cvtColor(x,cv2.COLOR_BGR2RGB)  #deal with channel order
    x = (x / 255)  # model expects 0..1
    x = torch.tensor(x).float()
    patches = x.unfold(0, 128, 128).unfold(1, 128, 128)
    patches = patches.contiguous().view(-1,3,128,128)

    return patches  # model expects a float 0..1

def processFrame(frame, model: UnetV1, device):
    x = preprocess(frame)

    with torch.no_grad():
        y = model.forward({'data': x.to(device)})['data']

    return postprocess(y.to('cpu'), frame)


def getUNetModelFromCheckpoint(checkpointPath, device):
    state_dict = torch.load(checkpointPath, map_location = 'cpu')['state_dict']
    clean_state_dict = OrderedDict()
    for key, value in state_dict.items():
        clean_state_dict[key.replace('Network.', '')] = value

    unetv1 = UnetV1(UnetV1Config(), (128, 128, 3))
    unetv1.load_state_dict(clean_state_dict)
    return unetv1.to(device)


def main():
    parser = argparse.ArgumentParser()                                               
    parser.add_argument('--input', '-i', type=str, required=True, help='filename of input video')
    parser.add_argument('--experiment', '-e', type=str, required=False, default = 'AlphaSegmentation', help='experiment name')
    parser.add_argument('--run', '-r', type=int, required=False, default = 0, help='run number')
    parser.add_argument('--checkpoint', '-c', type=str, required=True, help='checkpoint. the part of the filename that fills in the blank: \'epoch=____.ckpt\'')
    args = parser.parse_args()


    # load model
    checkpointPath = "./experiments/{}/run_{}/checkpoints/{}_epoch={}.ckpt".format(args.experiment, args.run, args.experiment, args.checkpoint)
    device = torch.device('cuda')
    #device = torch.device('cpu')
    unetv1 = getUNetModelFromCheckpoint(checkpointPath, device)

    #input
    input = cv2.VideoCapture(args.input)
    if not input.isOpened():
        print('Error: unable to find file ', args.input)
        return
    w = int(input.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(input.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = input.get(cv2.CAP_PROP_FPS)

    outputPath = os.path.join(os.path.dirname(args.input),
                                  'out_' + os.path.basename(args.input))
    out = cv2.VideoWriter(outputPath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
  
    while True:
        ret, inframe = input.read()
        if not ret: 
            break
        outframe = processFrame(inframe, unetv1, device)
        cv2.imshow('current frame', outframe)
        out.write(outframe)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    input.release()
    out.release()

    cv2.waitKey()

main()
