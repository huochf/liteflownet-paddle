import sys
sys.path.append('/home/aistudio/liteflownet')
import os
import argparse
import math
import numpy as np
from PIL import Image

import paddle.fluid as F
import paddle.fluid.layers as L
import paddle.fluid.dygraph as dg

from models import Network
from utils import visulize_flow


def prepare_data(first_image_path, second_image_path):
    tenFirst = np.array(Image.open(first_image_path).convert("RGB")).astype("float32")
    tenSecond = np.array(Image.open(second_image_path).convert("RGB")).astype("float32")
    mean = np.array([0.411618, 0.434631, 0.454253]).astype("float32")

    tenFirst = tenFirst / 255. - mean
    tenSecond = tenSecond / 255. - mean

    tenFirst = tenFirst.transpose((2, 0, 1))
    tenSecond = tenSecond.transpose((2, 0, 1))

    h, w = tenFirst.shape[1:]

    tenFirst = dg.to_variable(tenFirst)
    tenSecond = dg.to_variable(tenSecond)

    tenFirst = L.reshape(tenFirst, (1, 3, h, w))
    tenSecond = L.reshape(tenSecond, (1, 3, h, w))

    r_h, r_w = int(math.floor(math.ceil(h / 32.0) * 32.0)), int(math.floor(math.ceil(w / 32.0) * 32.0))
    tenFirst = L.image_resize(tenFirst, (r_h, r_w))
    tenSecond = L.image_resize(tenSecond, (r_h, r_w))
    return tenFirst, tenSecond, (h, w), (r_h, r_w)


def get_model(args):
    model = Network()
    state_dict, _ = F.load_dygraph(args.pretrained_model)
    model.load_dict(state_dict)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LiteFlownet Inference', add_help=False)
    parser.add_argument('--pretrained_model', default='./pretrained_models/network-default.pdparams',
        type=str, help="path to the pretrained model")
    parser.add_argument('--first', default="./images/first.png", type=str,
        help="path to the first image")
    parser.add_argument('--second', default="./images/second.png", type=str,
        help="path to the second image")
    parser.add_argument('--out', default="./images/flow.png", type=str, help="path to the output")
    args = parser.parse_args()

    with dg.guard():
        model = get_model(args)
        first, second, original_size, resized_size = prepare_data(args.first, args.second)
        flow = model(first, second)
        h, w = original_size
        r_h, r_w = resized_size
        flow = L.image_resize(flow, (h, w))

        flow[:, 0, :, :] *= float(w) / float(r_w)
        flow[:, 1, :, :] *= float(h) / float(r_h)
        flow = L.transpose(flow[0], (1, 2, 0)).numpy() # [h, w, 2]
        visulize_flow(flow, args.out)
