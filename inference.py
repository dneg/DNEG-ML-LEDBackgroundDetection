import numpy as np
import argparse, os, torch, cv2
from collections import OrderedDict

from src.Networks.UnetV1.UnetV1_config import UnetV1Config
from src.Networks.UnetV1.UnetV1_component import UnetV1

TS =  512   #tile size
OV =   64   #overlap
STR = TS-OV #stride
DOWNSAMPLE = 1  #number of times to halve the input dimensions

def calcPadding(imgShape):
    H, W, _ = imgShape
    dims = np.array([H, W])

    remainders = np.mod(dims, STR)
    pad = np.mod(remainders * -1 + OV, STR)
    pad[pad<OV]+=STR
    pad = (pad / 2).astype('int64')
    return tuple(pad) 

# takes the result of a batch (B,C,H,W) image and stitches it back into a cv2 image (H,W,C) ()
def postprocess(x, original):
    hpad, wpad = calcPadding(original.shape)
    H, W,_ = original.shape #width and height of output

    #x starts out (B,1,TH,TW)
    x = x.squeeze() # (B,TH,TW)
    toTrim = int(OV/2)
    x = x[:,toTrim:STR+toTrim,toTrim:STR+toTrim]  #trim off overlap

    numTilesH, numTilesW = int((H+hpad*2)/STR), int((W+wpad*2)/STR)
    x = x.view(numTilesH, numTilesW, STR, STR)  # split B: (#H, #W, TH, TW)
    x = x.permute((0,2,1,3)) #(#H,TH,#W,TW)
    x = x.contiguous().view(numTilesH*STR, numTilesW*STR)  #H, W

    #crop out any extra padding
    toTrimH = int((x.shape[0] - H)/2)
    toTrimW = int((x.shape[1] - W)/2)
    x = x[toTrimH:H+toTrimH, toTrimW:W+toTrimW]


    if False:
        #RETURN BW mask
        x[x > 0.5] = 255
        x = np.dstack((x,x,x))  # as three channels
    elif False:
        #RETURN grayscale mask
        x = x * 255
        x = np.dstack((x,x,x))  # as three channels
    else:
        #WITH BACKGROUND
        x = x.numpy()
        xx, yy = np.mgrid[0:original.shape[0],0:original.shape[1]]
        bkgd = ((xx / 16).astype('int') + (yy / 16).astype('int')) % 2 * 0x33 + 0x66
        imgpart = original * x[:,:,None]
        bkgdpart = bkgd * (x * -1 + 1)
        x = imgpart + bkgdpart[:,:,None]

    return x.astype('uint8')

#takes a  cv2 image (H,W,C) with channels = (BGR)
#returns a batch of tensors to be (B,C,H,W) with channels = (RGB)
def preprocess(x):
    hpad, wpad = calcPadding(x.shape)

    x = np.pad(x, (((hpad, hpad),(wpad, wpad), (0,0))), 'constant')

    #Frame from cv2 is (H,W,C) with channels = (BGR)
    #we want a batch of tensors to be (B,C,H,W) with channels = (RGB)
    x = cv2.cvtColor(x,cv2.COLOR_BGR2RGB)  #deal with channel order
    x = (x / 255)  # model expects 0..1
    x = torch.tensor(x).float()
    patches = x.unfold(0, TS, STR).unfold(1, TS, STR)
    patches = patches.contiguous().view(-1,3,TS,TS)

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

    unetv1 = UnetV1(UnetV1Config(), (TS, TS, 3))
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
    unetv1 = getUNetModelFromCheckpoint(checkpointPath, device)

    if args.input.endswith('.mp4'):
        #input is video
        input = cv2.VideoCapture(args.input)
        if not input.isOpened():
            print('Error: unable to find file ', args.input)
            return
        w = int(input.get(cv2.CAP_PROP_FRAME_WIDTH)) >> DOWNSAMPLE
        h = int(input.get(cv2.CAP_PROP_FRAME_HEIGHT)) >> DOWNSAMPLE
        fps = input.get(cv2.CAP_PROP_FPS)
    
        outputPath = os.path.join(os.path.dirname(args.input),
                                      'out_' + os.path.basename(args.input))
        out = cv2.VideoWriter(outputPath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
      
        while True:
            ret, inframe = input.read()
            if not ret: 
                break
            inframe = cv2.resize(inframe, (w, h), interpolation = cv2.INTER_AREA)
            outframe = processFrame(inframe, unetv1, device)
            cv2.imshow('current frame', outframe)
            out.write(outframe)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    
        input.release()
        out.release()
        cv2.waitKey()

    else:
        #input is an image
        input = cv2.imread(args.input)
        cv2.imshow('in', input)
        pre = preprocess(input)
        output = pre[:,0,:,:].squeeze()
        post = postprocess(output,input)
        cv2.imshow('out', post)
        cv2.waitKey()

main()
