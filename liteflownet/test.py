import sys
sys.path.append('.')
import numpy as np
import paddle.fluid.dygraph as dg

from models import Features, Matching, Subpixel, Regularization, Network


if __name__ == '__main__':
    # moduleFeatures = Features()
    # for k, v in moduleFeatures.state_dict().items():
    #     print(k + ': ' + str(v.shape))
    
    # netMatching = [Matching(intLevel) for intLevel in [2, 3, 4, 5, 6]]

    # print()
    # for m in netMatching:
    #     for k, v in m.state_dict().items():
    #         print(k + ": " + str(v.shape))
    #     print()
    
    # netSubpixel = [Subpixel(intLevel) for intLevel in [2, 3, 4, 5, 6]]

    # print()
    # for s in netSubpixel:
    #     for k, v in s.state_dict().items():
    #         print(k + ': ' + str(v.shape))
    #     print()
    
    # netRegularization = [Regularization(intLevel) for intLevel in [2, 3, 4, 5, 6]]
    # print()
    # for r in netRegularization:
    #     for k, v in r.state_dict().items():
    #         print(k + ": " + str(v.shape))
    #     print()
    
    # print("----------------------------------------------------------")
    # flownet = Network()
    # for k, v in flownet.state_dict().items():
    #     print(k + ": " + str(v.shape))

    with dg.guard():
        flownet = Network()
        flownet.eval()
        tenFirst = dg.to_variable(np.zeros((1, 3, 1024, 1024)).astype("float32"))
        tenSecond = dg.to_variable(np.zeros((1, 3, 1024, 1024)).astype("float32"))
        out = flownet(tenFirst, tenSecond)
        print(out.shape)












