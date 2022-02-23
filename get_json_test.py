import cv2
import os,shutil
import json
import tqdm
root_dir='/data/tianhao.lu/data/Iris/DeepsnakeDatasets/NewThreeDatasets/Off_angle'
split='test'
imgs_path1 = r'{}/{}/iris_edge'.format(root_dir,split)
imgs_path2 = r'{}/{}/pupil_edge'.format(root_dir,split)
imgs_path3=r'{}/{}/image'.format(root_dir,split)
seg_path=r'{}/{}/SegmentationClass'.format(root_dir,split)
json_path = r'./0.json'

try:
    with tqdm.tqdm(range(len(os.listdir(imgs_path1)))) as tqdm_range:
        for _,img_name in zip(tqdm_range,os.listdir(imgs_path1)):
            img_path1 = os.path.join(imgs_path1,img_name)
            img_path2 = os.path.join(imgs_path2,img_name)
            img_path3=os.path.join(imgs_path3,img_name.replace('png','jpg'))
            seg_path4=os.path.join(seg_path,img_name)
            # print(img_path3)
            os.makedirs('tmp/Anno/{}/bp/'.format(split),exist_ok=True)
            new_js_pt = os.path.join('tmp/Anno/{}/bp/'.format(split),img_name.replace('png','json'))
            output_path='tmp/leftImg8bit/{}/bp'.format(split)
            os.makedirs(output_path,exist_ok=True)
            new_img_pt=os.path.join(output_path,img_name.replace('png','jpg'))

            output_seg_path='tmp/leftImg8bit/{}/SegmentationClass'.format(split)
            os.makedirs(output_seg_path,exist_ok=True)
            new_seg_img_pt=os.path.join(output_seg_path,img_name)

            # print(new_js_pt)
            if(not os.path.exists(seg_path4)):
                print(seg_path4+' not exists')
                continue

                
            shutil.copy(json_path,new_js_pt)
            shutil.copy(img_path3,new_img_pt)
            # print(new_seg_img_pt)
            shutil.copy(seg_path4,new_seg_img_pt)

            ALL = []

            img1 = cv2.imread(img_path1, 0)
            # image,contours, _ = cv2.findContours(img1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours, _ = cv2.findContours(img1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            C = []
            try:
                for con in contours[0]:
                    C.append(con[0].tolist())

                f = open(json_path,'r')
                datas = json.load(f)

                data = datas[0]
                data["components"][0]['poly'] = C
                data["img_path"] = os.path.join('tmp/leftImg8bit/{}/bp'.format(split),img_name.replace('png','jpg')) #/root/LIKE/虹膜/snake/tmp/leftImg8bit/train/bp
                data["label"] = 'bp'
                data["image_id"] = img_name.split('.')[0]
                ALL.append(data)

                img2 = cv2.imread(img_path2,0)
                img2[img2<100] = 0
                draw = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)

                contours, _  = cv2.findContours(img2, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE ) #_, contours, hierarchy
                # cv2.drawContours(draw, contours_, -1, (0, 0, 255), 2)
                # cv2.imshow('test',draw)
                # cv2.waitKey()

                C_ = []
                for con in contours[0]:
                    C_.append(con[0].tolist())
                f = open(json_path, 'r')
                datas = json.load(f)
                data = datas[0]
                data["components"][0]['poly'] = C_
                data["img_path"] = os.path.join('tmp/leftImg8bit/{}/bp'.format(split), img_name.replace('png', 'jpg'))
                data["label"] = 'bp_n'
                data["image_id"] = img_name.split('.')[0]
                ALL.append(data)

                with open(new_js_pt, "w") as jsonFile:
                    json.dump(ALL, jsonFile,ensure_ascii=False)
            except:
                print(img_name)
except KeyboardInterrupt:
    tqdm_range.close()
tqdm_range.close()

# path = r'tmp/leftImg8bit/train/bp'
# for name in os.listdir(path):
#     os.rename(os.path.join(path,name),os.path.join(path,name.replace('JPG','jpg')))
