import numpy as np
import argparse, os, torch, cv2
from collections import OrderedDict

from src.Networks.UnetV1.UnetV1_config import UnetV1Config
from src.Networks.UnetV1.UnetV1_component import UnetV1
from src.Networks.Resnet34Seg.Resnet34Seg_config import Resnet34SegConfig
from src.Networks.Resnet34Seg.Resnet34Seg_component import Resnet34Seg

TS =  512   #tile size
OV =   32   #overlap
STR = TS-OV #stride
args = None

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
    global args
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

    if args.mode == 'bw':
        #BW mask
        x[x > 0.5] = 255
        x = np.dstack((x,x,x))  # as three channels
    elif args.mode == 'grayscale':
        #grayscale mask
        x = x * 255
        x = np.dstack((x,x,x))  # as three channels
    elif args.mode in ['checkerbg', 'compare']: 
        #With checkerboard background
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
    batches = x.split(8, dim=0)

    y = []
    for b in batches:
        with torch.no_grad():
            result = model.forward({'data': b.to(device)})
        y.append(result['data'].to('cpu'))

    return postprocess(torch.concat(y), frame)


def getUNetModelFromCheckpoint(checkpointPath, device):
    model = torch.load(checkpointPath, map_location='cpu')
    state_dict = model['state_dict']
    clean_state_dict = OrderedDict()
    for key, value in state_dict.items():
        newkey = key.replace('Network.', '')
        #newkey = newkey.replace('.weight', '.conv.weight')
        #newkey = newkey.replace('.bias', '.conv.bias')
        clean_state_dict[newkey] = value

    modelIsResnet = any(['res_block' in x for x in state_dict.keys()])

    if modelIsResnet:
        resnetseg = Resnet34Seg(Resnet34SegConfig(NumOutputs=1), (TS, TS, 3))
        resnetseg.load_state_dict(clean_state_dict)
        return resnetseg.to(device)
    else:
        unetv1 = UnetV1(UnetV1Config(), (TS, TS, 3))
        unetv1.load_state_dict(clean_state_dict)
        return unetv1.to(device)

def main():
    global args
    parser = argparse.ArgumentParser()                                               
    parser.add_argument('--input', '-i', type=str, required=True, help='filename of input video')
    parser.add_argument('--experiment', '-e', type=str, required=False, default = 'AlphaSegmentation', help='experiment name')
    parser.add_argument('--run', '-r', type=int, required=False, default = 0, help='run number')
    parser.add_argument('--checkpoint', '-c', type=str, required=True, help='checkpoint. the part of the filename that fills in the blank: \'epoch=____.ckpt\'')
    parser.add_argument('--mode', '-m', type=str, default = 'checkerbg', help='either: bw, grayscale, or checkerbg (default)')
    parser.add_argument('--downsample', '-d', type=int, default = 0, help='numtimes to halve the width/height of input')
    args = parser.parse_args()

    assert args.mode in ['bw', 'grayscale', 'checkerbg', 'compare'], 'Error: Unrecognized output mode, must be bw, grayscale, checkerbg, or compare'
    assert os.path.exists(args.input), 'Error: unable to find file ' + args.input

    # calculate output filename
    outputDir = "./experiments/{}/run_{}/out/".format(args.experiment, args.run)
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    outputPath = os.path.join(outputDir, '{}{}_{}'.format(args.mode, args.checkpoint, os.path.basename(args.input)))


    # load model
    checkpointPath = "./experiments/{}/run_{}/checkpoints/{}_epoch={}.ckpt".format(args.experiment, args.run, args.experiment, args.checkpoint)
    device = torch.device('cuda')
    unetv1 = getUNetModelFromCheckpoint(checkpointPath, device)

    if args.input.endswith('.mp4'):
        #input is video
        input = cv2.VideoCapture(args.input)
        assert input.isOpened(), 'Error: unable to open file ' + args.input
        w = int(input.get(cv2.CAP_PROP_FRAME_WIDTH)) >> args.downsample
        h = int(input.get(cv2.CAP_PROP_FRAME_HEIGHT)) >> args.downsample
        fps = input.get(cv2.CAP_PROP_FPS)
        outh = h*2 if args.mode == 'compare' else h
        out = cv2.VideoWriter(outputPath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,outh))
      
        while True:
            ret, inframe = input.read()
            if not ret: 
                break
            inframe = cv2.resize(inframe, (w, h), interpolation = cv2.INTER_AREA)
            outframe = processFrame(inframe, unetv1, device)
            if args.mode == 'compare':
                outframe = np.concatenate([inframe, outframe], axis=0)
            cv2.imshow('current frame', outframe)
            out.write(outframe)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    
        input.release()
        out.release()
        cv2.waitKey()

    else:
        #input is an image
        inframe = cv2.imread(args.input)
        H, W, _ = inframe.shape
        w = W >> args.downsample
        h = H >> args.downsample
        inframe = cv2.resize(inframe, (w, h), interpolation = cv2.INTER_AREA)
        outframe = processFrame(inframe, unetv1, device)
        cv2.imshow('out', cv2.resize(outframe, (1024, int(h*1024/w)), interpolation = cv2.INTER_AREA))
        cv2.imwrite(outputPath, outframe)
        cv2.waitKey()

main()
