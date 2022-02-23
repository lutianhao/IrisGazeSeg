import torch.utils.data as data
import glob
import os
import cv2
import numpy as np
from lib.utils.snake import snake_config
from lib.utils import data_utils
from lib.config import cfg
import tqdm
import torch
from lib.networks import make_network
from lib.utils.net_utils import load_network
from lib.visualizers import make_visualizer
from PIL import Image
from termcolor import colored

class Dataset(data.Dataset):
    def __init__(self):
        super(Dataset, self).__init__()

        print(cfg.demo_path)
        if os.path.isdir(cfg.demo_path):
            self.imgs = glob.glob(os.path.join(cfg.demo_path, '*'))
        elif os.path.exists(cfg.demo_path):
            self.imgs = [cfg.demo_path]
        else:
            raise Exception('NO SUCH FILE')

    def normalize_image(self, inp):
        inp = (inp.astype(np.float32) / 255.)
        inp = (inp - snake_config.mean) / snake_config.std
        inp = inp.transpose(2, 0, 1)
        return inp

    def __getitem__(self, index):
        img_name = self.imgs[index]
        img = cv2.imread(img_name)
        # img = cv2.resize(img,(512,512))
        # print(img)
        image = Image.open(img_name).convert("RGB")
        image = np.array(image, dtype=np.uint8)

        
        x = 32
        # input_w = (int(width / 1.) | (x - 1)) + 1
        # input_h = (int(height / 1.) | (x - 1)) + 1

        input_w = 512
        input_h = 512

        # trans_input = data_utils.get_affine_transform(center, scale, 0, [input_w, input_h])
        # inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)
        inp = cv2.resize(img, (input_w, input_h), interpolation = cv2.INTER_LINEAR)

        width, height = img.shape[1], img.shape[0]
        center = np.array([width // 2, height // 2])
        scale = np.array([width, height])
        # print(inp)
        inp = self.normalize_image(inp)
        ret = {'inp': inp,'image':image}
        meta = {'center': center, 'scale': scale, 'test': '', 'ann': ''}
        ret.update({'meta': meta})

        return ret

    def __len__(self):
        return len(self.imgs)


def test():
    network = make_network(cfg).cuda()
    cfg.demo_path='tmp/leftImg8bit/test/bp'
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    network.eval()

    # seg_net = torch.load('net.pt', map_location='cuda')['model']

    dataset = Dataset()
    visualizer = make_visualizer(cfg)
    for batch in tqdm.tqdm(dataset):
        try:
            batch['inp'] = torch.FloatTensor(batch['inp'])[None].cuda()
            # print(batch['meta'].keys()) # dict_keys(['center', 'scale', 'test', 'ann'])
            with torch.no_grad():
                output,seg_out = network(batch['inp'], batch)
                # print(seg_out)
                # print(seg_out.shape)  # (512, 512)
            visualizer.visualize(output,batch,network)
        except:
            print(colored("wrong Pic!","red"))
