# Face alignment and crop demo
# Uses MTCNN or FaceBoxes as a face detector;
# Support different backbones, include PFLD, MobileFaceNet, MobileNet;
# Cunjian Chen (ccunjian@gmail.com), Feb. 2021

from __future__ import division
import argparse
import torch
import os
import cv2
import numpy as np
from common.utils import BBox,drawLandmark,drawLandmark_multiple
from models.basenet import MobileNet_GDConv
from models.pfld_compressed import PFLDInference
from models.mobilefacenet import MobileFaceNet
from FaceBoxes import FaceBoxes
from Retinaface import Retinaface
from PIL import Image
import matplotlib.pyplot as plt
from MTCNN import detect_faces
import glob
import time
from utils.align_trans import get_reference_facial_points, warp_and_crop_face

parser = argparse.ArgumentParser(description='PyTorch face landmark')
# Datasets
parser.add_argument('--backbone', default='MobileFaceNet', type=str,
                    help='choose which backbone network to use: MobileNet, PFLD, MobileFaceNet')
parser.add_argument('--detector', default='Retinaface', type=str,
                    help='choose which face detector to use: MTCNN, FaceBoxes, Retinaface')

args = parser.parse_args()
mean = np.asarray([ 0.485, 0.456, 0.406 ])
std = np.asarray([ 0.229, 0.224, 0.225 ])

crop_size= 112
scale = crop_size / 112.
reference = get_reference_facial_points(default_square = True) * scale

if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'

def load_model():
    if args.backbone=='MobileNet':
        model = MobileNet_GDConv(136)
        model = torch.nn.DataParallel(model)
        # download model from https://drive.google.com/file/d/1Le5UdpMkKOTRr1sTp4lwkw8263sbgdSe/view?usp=sharing
        checkpoint = torch.load('checkpoint/mobilenet_224_model_best_gdconv_external.pth.tar', map_location=map_location)
        print('Use MobileNet as backbone')
    elif args.backbone=='PFLD':
        model = PFLDInference() 
        # download from https://drive.google.com/file/d/1gjgtm6qaBQJ_EY7lQfQj3EuMJCVg9lVu/view?usp=sharing
        checkpoint = torch.load('checkpoint/pfld_model_best.pth.tar', map_location=map_location)
        print('Use PFLD as backbone') 
        # download from https://drive.google.com/file/d/1T8J73UTcB25BEJ_ObAJczCkyGKW5VaeY/view?usp=sharing
    elif args.backbone=='MobileFaceNet':
        model = MobileFaceNet([112, 112],136)   
        checkpoint = torch.load('checkpoint/mobilefacenet_model_best.pth.tar', map_location=map_location)      
        print('Use MobileFaceNet as backbone')         
    else:
        print('Error: not suppored backbone')    
    model.load_state_dict(checkpoint['state_dict'])
    return model

if __name__ == '__main__':
    if args.backbone=='MobileNet':
        out_size = 224
    else:
        out_size = 112 
    model = load_model()
    model = model.eval()
    filenames=glob.glob("samples/12--Group/*.jpg")
    for imgname in filenames:
        print(imgname)
        img = cv2.imread(imgname)
        org_img = Image.open(imgname)
        height,width,_=img.shape
        if args.detector=='MTCNN':
            # perform face detection using MTCNN
            image = Image.open(imgname)
            faces, landmarks = detect_faces(image)
        elif args.detector=='FaceBoxes':
            face_boxes = FaceBoxes()
            faces = face_boxes(img)
        elif args.detector=='Retinaface':
            retinaface=Retinaface.Retinaface()    
            faces = retinaface(img)            
        else:
            print('Error: not suppored detector')        
        ratio=0
        if len(faces)==0:
            print('NO face is detected!')
            continue
        for k, face in enumerate(faces): 
            if face[4]<0.9: # remove low confidence detection
                continue
            x1=face[0]
            y1=face[1]
            x2=face[2]
            y2=face[3]
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            size = int(min([w, h])*1.2)
            cx = x1 + w//2
            cy = y1 + h//2
            x1 = cx - size//2
            x2 = x1 + size
            y1 = cy - size//2
            y2 = y1 + size

            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)

            edx = max(0, x2 - width)
            edy = max(0, y2 - height)
            x2 = min(width, x2)
            y2 = min(height, y2)
            new_bbox = list(map(int, [x1, x2, y1, y2]))
            new_bbox = BBox(new_bbox)
            cropped=img[new_bbox.top:new_bbox.bottom,new_bbox.left:new_bbox.right]
            if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
                cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)            
            cropped_face = cv2.resize(cropped, (out_size, out_size))

            if cropped_face.shape[0]<=0 or cropped_face.shape[1]<=0:
                continue
            test_face = cropped_face.copy()
            test_face = test_face/255.0
            if args.backbone=='MobileNet':
                test_face = (test_face-mean)/std
            test_face = test_face.transpose((2, 0, 1))
            test_face = test_face.reshape((1,) + test_face.shape)
            input = torch.from_numpy(test_face).float()
            input= torch.autograd.Variable(input)
            start = time.time()
            if args.backbone=='MobileFaceNet':
                landmark = model(input)[0].cpu().data.numpy()
            else:
                landmark = model(input).cpu().data.numpy()
            end = time.time()
            print('Time: {:.6f}s.'.format(end - start))
            landmark = landmark.reshape(-1,2)
            landmark = new_bbox.reprojectLandmark(landmark)
            img = drawLandmark_multiple(img, new_bbox, landmark)
            # crop and aligned the face
            lefteye_x=0
            lefteye_y=0
            for i in range(36,42):
                lefteye_x+=landmark[i][0]
                lefteye_y+=landmark[i][1]
            lefteye_x=lefteye_x/6
            lefteye_y=lefteye_y/6
            lefteye=[lefteye_x,lefteye_y]

            righteye_x=0
            righteye_y=0
            for i in range(42,48):
                righteye_x+=landmark[i][0]
                righteye_y+=landmark[i][1]
            righteye_x=righteye_x/6
            righteye_y=righteye_y/6
            righteye=[righteye_x,righteye_y]  

            nose=landmark[33]
            leftmouth=landmark[48]
            rightmouth=landmark[54]
            facial5points=[righteye,lefteye,nose,rightmouth,leftmouth]
            warped_face = warp_and_crop_face(np.array(org_img), facial5points, reference, crop_size=(crop_size, crop_size))
            img_warped = Image.fromarray(warped_face)
            # save the aligned and cropped faces
            img_warped.save(os.path.join('results_aligned', os.path.basename(imgname)[:-4]+'_'+str(k)+'.png'))  
            #img = drawLandmark_multiple(img, new_bbox, facial5points)  # plot and show 5 points   
        # save the landmark detections 
        cv2.imwrite(os.path.join('results',os.path.basename(imgname)),img)

