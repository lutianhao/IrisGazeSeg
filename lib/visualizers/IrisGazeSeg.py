from lib.utils import img_utils, data_utils
from lib.utils.snake import snake_config
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import pycocotools.coco as coco
from lib.config import cfg
from lib.datasets.dataset_catalog import DatasetCatalog
import os
import cv2
from itertools import cycle
from seg_demo import demo_
import os
from termcolor import colored

mean = snake_config.mean
std = snake_config.std


class Visualizer:
    def __init__(self):
        self.coco = None

    def visualize_ex(self, output, batch,seg_net):
        inp = img_utils.bgr_to_rgb(img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0))
        # print("inp shape: {}".format(inp.shape))  # torch.size([512,512,3])
        detection = output['detection']
        score = detection[:, 4].detach().cpu().numpy()
        label = detection[:, 5].detach().cpu().numpy().astype(int)
        cp_ind = output['cp_ind'].detach().cpu().numpy().astype(int)
        py = output['py'][-1].detach().cpu().numpy() * snake_config.down_ratio

        Num_label = np.unique(label)
        tmp_score = []
        tmp_label = []
        tmp_py = []
        while True:
            for i in range(len(Num_label)):
                tmp_score.append(score[np.nonzero(label == i)][0])
                tmp_label.append(label[np.nonzero(label == i)][0])
                tmp_py.append(py[np.nonzero(label == i)][0])
            #     for j in range(len(label)):
            #         if j == 0 :
            #             tmp_score.append(score[j])
            #             tmp_label.append(label[j])
            #             tmp_py.append(py[j])
            #         elif label[j] != i:
            #             break
            score = np.array(tmp_score)
            label = np.array(tmp_label)
            py = np.array(tmp_py)
            if len(score) == 2 :
                cp_ind = np.array([0,1])
                break
            elif len(score) < 2 :
                break

        if len(py) == 0 or len(py) != 2 :
            print(colored("Wrong edge Number!","red"))
            # return

        ct_ind = np.unique(cp_ind)
        score = score[ct_ind]
        label = label[ct_ind]

        ind_group = [np.argwhere(ct_ind[i] == cp_ind).ravel() for i in range(len(ct_ind))]
        py = [[py[ind] for ind in inds] for inds in ind_group]

        fig, ax = plt.subplots(1, figsize=(20, 10))
        fig.tight_layout()
        ax.axis('off')
        ax.imshow(inp)

        colors = np.array([
            [31, 119, 180],
            [255, 127, 14],
            [46, 160, 44],
            [214, 40, 39],
            [148, 103, 189],
            [140, 86, 75],
            [227, 119, 194],
            [126, 126, 126],
            [188, 189, 32],
            [26, 190, 207]
        ]) / 255.
        # colors = cycle(colors)
        num=2
        # for i in range(len(py)):
        for i in range(num):
            color = colors[np.random.randint(len(colors))]
            # color = next(colors).tolist()
            for poly in py[i]:
                poly = np.append(poly, [poly[0]], axis=0)
                ax.plot(poly[:, 0], poly[:, 1], color=color, linewidth=3)

        # plt.show()
        # img_name = str(len(os.listdir('/root/Downloads/HM/snake(data2)/out')))+'.jpg'
        # plt.savefig(os.path.join('/root/Downloads/HM/snake(data2)/out',img_name))
        # plt.close()

        P = py[0]
        P = np.array(P).astype(int)
        A = inp.numpy()
        A = cv2.cvtColor(A,cv2.COLOR_RGB2BGR)
        A = A * 255.0
        A = A.astype(np.uint8)
        B = A.copy()
        A_ = A.copy()

        for i,pp in enumerate(py):
            pp = np.array(pp).astype(int)
            draw = cv2.drawContours(A_,pp,-1,(0,0,255),2)
            if i == 1:break

        res = cv2.fillConvexPoly(A, P, (255, 255, 255))
        res = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
        res[res != 255] = 0
        # print(res.shape)
        # res = cv2.cvtColor(res,cv2.COLOR_GRAY2BGR)
        # 输出mask图像
        out_path='out'
        os.makedirs(out_path,exist_ok=True)
        seg_img = cv2.add(B, np.zeros(np.shape(B),dtype=np.uint8), mask=res)
        img_name = str(len(os.listdir(out_path))) + '.jpg'
        cv2.imwrite(os.path.join('test1',img_name),seg_img)

        # 输出定位图像
        pred,pred_color = demo_(batch,seg_net)
        
        index=len(os.listdir(out_path))
        img_name = str(index)+'_mask.jpg'
        # print(py)
        # print(pred.shape) # (512,512)
        py,seg_out=self.post_process(inp,py, pred)
        # seg_out = np.squeeze(seg_out)
        # print(seg_out.shape) # (512,512)

        colormap = [[0, 0, 0], [220, 20, 60], [119, 11, 32]]
        ndcolormap = np.array(colormap)
        post_out= ndcolormap[seg_out].astype(np.uint8)

        cv2.imwrite(os.path.join(out_path,img_name),seg_out*255)
        # cv2.imwrite(os.path.join(out_path,img_name),pred*255)

        new = cv2.addWeighted(draw, 1., post_out, 0.7, 0)

        img_name = str(index)+'.jpg'
        # cv2.imwrite(os.path.join(out_path,img_name),new)
        cv2.imwrite(os.path.join(out_path,img_name),new)
        # # cv2.imshow('test',new)
        # # cv2.waitKey()



    def visualize_training_box(self, output, batch):
        inp = img_utils.bgr_to_rgb(img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0))
        box = output['detection'][:, :4].detach().cpu().numpy() * snake_config.down_ratio
        # box = output['cp_box'][:, :4].detach().cpu().numpy() * snake_config.down_ratio

        _, ax = plt.subplots(1)
        ax.imshow(inp)
        n = len(box)
        for i in range(n):
            x_min, y_min, x_max, y_max = box[i]
            ax.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min])
        plt.show()

    def visualize(self, output, batch,seg_net):
        self.visualize_ex(output, batch,seg_net)
        # self.visualize_training_box(output, batch)


    def post_process(self,img,location_res, segmentation_res):
        #判断是否内外边界都有：
        # print(len(location_res))
        # print(len(location_res[0]))
        # print(len(location_res[1]))
        # # print(location_res[0][0].shape)
        # # print(location_res[1][0].shape)
        # # print(location_res[0][0])
        # # print(location_res[1][0])
        if len(location_res[0][0])!=0 and len(location_res[1][0]!=0):
            # 判断内边界是否在外边界之内
            flag = False #判断内外边界是否出错，如果没出错则继续用来约束分割区域
            # Iris_edge_mask = np.zeros((img.shape[0], img.shape[1], 1), dtype="uint8")
            # cv2.fillPoly(Iris_edge_mask, np.int32([location_res[0][0]]), 1)
            # for point_location in location_res[1][0]:
            #     # print(point_location)
            #     x=int(point_location[0])
            #     y=int(point_location[1])
            #     # print(x,y)
            #     # print(Iris_edge_mask[x][y])
            #     if Iris_edge_mask[x][y][0] == 0:
            #         print(colored('location warning!!','red'))
            #         flag = True
            #         break
            if flag == False:
                threshold = 0.2  #设置阈值，判断是否根据定位边界修改分割区域
                Num_total = cv2.countNonZero(segmentation_res) #统计有效区域像素点总个数
                Num_outside = 0  # 记录超出定位边界的像素点个数
                Iris_pupil_mask = np.zeros((img.shape[0],img.shape[1],1), dtype="uint8")
                cv2.fillPoly(Iris_pupil_mask, np.int32([location_res[0][0]]), 1)
                cv2.fillPoly(Iris_pupil_mask, np.int32([location_res[1][0]]), 0)
                segmentation_resNew = segmentation_res.copy()
                # print(segmentation_resNew.shape)
                for x in range(segmentation_resNew.shape[0]):
                    for y in range(segmentation_resNew.shape[1]):
                        # print(x,y)
                        if Iris_pupil_mask[x][y] == 0 and segmentation_resNew[x][y] != 0:
                            # print('hello')
                            Num_outside +=1
                            segmentation_resNew[x][y] = 0
                if Num_total!=0 and Num_outside / Num_total < threshold:
                    print('output new')
                    return location_res, segmentation_resNew
                else:
                    return location_res, segmentation_res
                return location_res, segmentation_res

            else:
                return location_res , segmentation_res

