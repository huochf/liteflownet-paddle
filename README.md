# LiteFlowNet-Paddle

### About the code

Flow estimation model for [LiteFlowNet](https://arxiv.org/pdf/1805.07036.pdf), this code is modified from [this implementation](https://github.com/sniklaus/pytorch-liteflownet).
This project is conducted on [AI Studio](https://aistudio.baidu.com/bjcpu/user/229528/1354069/notebooks/1354069.ipynb).

### Pretrained models

these models are converted from [pytorch-liteflownet](https://github.com/sniklaus/pytorch-liteflownet).


</ul>
<table>
<thead>
<tr>
<th align="center"></th>
<th align="center">KITTI12 Testing Set (Out-Noc)</th>
<th align="center">KITTI15 Testing Set (Fl-all)</th>
<th align="center">Model Size (M)</th>
<th align="center">Paddle Model</th>
</tr>
<tr>
<td align="center"><strong>LiteFlowNet (CVPR18)</strong></td>
<td align="center"><strong>3.27%</strong></td>
<td align="center"><strong>9.38%</strong></td>
<td align="center"><strong>5.37</strong></td>
  <td align="center"><a href="https://aistudio.baidu.com/aistudio/datasetdetail/66040">model</a></td>
</tr>    
</tbody></table>

If this work is useful for you, please cite:

@inproceedings{Hui_CVPR_2018, \
author = {Tak-Wai Hui and Xiaoou Tang and Chen Change Loy}, \
title = {{LiteFlowNet}: A Lightweight Convolutional Neural Network for Optical Flow Estimation}, \
booktitle = {IEEE Conference on Computer Vision and Pattern Recognition}, \
year = {2018} \
}
