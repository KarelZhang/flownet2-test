import argparse
import os
from ..net import Mode
from .flownet2 import FlowNet2
from .dataloader import Dataloader

FLAGS = None


def main():

    # prepare data
    data_root= ''
    filenames_file = 'test_kitti_2015_flow.txt'
    data = Dataloader(data_root, filenames_file)

    input_a = data.left_path
    input_b = data.right_path

    out = './result/kitti2015'

    # # Create a new network
    net = FlowNet2(mode=Mode.TEST)

    # # Train on the data
    net.test_kitti(
        checkpoint='./checkpoints/FlowNet2/flownet-2.ckpt-0',
        input_a_path=input_a,
        input_b_path=input_b,
        out_path=out
    )


if __name__ == '__main__':
    main()
