#!/usr/bin/python

# from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from cv_bridge.boost.cv_bridge_boost import getCvType
from datetime import datetime
from geometry_msgs.msg import Twist, Point
from model_fpn import D2N
# from p2os_msgs.msg import MotorState
from scipy.ndimage import filters
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import String
from threading import Thread
from torch.autograd import Variable
from torchvision.utils import save_image
import argparse, time
import cv2
# import cv2.cv
import message_filters
import numpy as np
import os, sys
import roslib
import rospy
import timeit
import torch, time
# from pcl import *

class Estimator:
    def __init__(self):
        self.args = self.parse_args()
        if torch.cuda.is_available() and not self.args.cuda:
            print("WARNING: You might want to run with --cuda")
        print('Initializing model...')
        self.d2n = D2N(fixed_feature_weights=False)
        if self.args.cuda:
            self.d2n = self.d2n.cuda()  
        print('Done!')        
        load_name = os.path.join(self.args.model_path)
        print("loading checkpoint %s" % (load_name))
        state = self.d2n.state_dict()
        checkpoint = torch.load(load_name)
        checkpoint = {k: v for k, v in checkpoint['model'].items() if k in state}
        state.update(checkpoint)
        self.d2n.load_state_dict(state)
        if 'pooling_mode' in checkpoint.keys():
            POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))
        del checkpoint
        torch.cuda.empty_cache()
        self.d2n.eval()
        self.img = Variable(torch.FloatTensor(1), volatile=True)
        
        rospy.init_node('estimate', anonymous=True)
        self.pub_cloud = rospy.Publisher('/pcd_normals', PointCloud2,queue_size=1)
        self.cam_info = message_filters.Subscriber(self.args.intopic+'camera_info', CameraInfo)
        self.depth_sub = message_filters.Subscriber(self.args.intopic+'image_raw', Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.cam_info, self.depth_sub], 1, 1)
        self.ts.registerCallback(self.callback)
        print('evaluating started, waiting topics...')


    def parse_args(self):
        """
        Parse input arguments
        """
        parser = argparse.ArgumentParser(description='Normal image estimation from ToF depth image')
        parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        default=True,
                        action='store_true')
        parser.add_argument('--num_workers', dest='num_workers',
                        help='num_workers',
                        default=1, type=int)   
        parser.add_argument('--input_image_path', dest='input_image_path',
                        help='path to a single input image for evaluation',
                        default='/home/cuda/TofNet/training_images/normal.png', type=str)
        parser.add_argument('--eval_folder', dest='eval_folder',
                        help='evaluate only one image or the whole folder',
                        default=False, type=bool)
        parser.add_argument('--model_path', dest='model_path',
                        help='path to the model to use',
                        default='/home/cuda/catkin_ws/src/ToFNest_rt/src/saved_models/d2n_1_9_v35.pth', type=str)
        parser.add_argument('--input_topic', dest='intopic',
                        help='name of the topicfrom where the program should read the depth image and the camera info',
                        default='/pico_zense/depth/', type=str)

        args = parser.parse_args()
        return args




    def publish_pcd(self,camera_info, depth, normals):
        print("creating cloud...")
        fx = camera_info.K[0]
        fy = camera_info.K[4]
        x0 = camera_info.K[2]
        y0 = camera_info.K[5]
        scalingFactor = 1000
        points=[]
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1),
                    PointField('r', 12, PointField.INT8, 1),
                    PointField('g', 13, PointField.INT8, 1),
                    PointField('b', 14, PointField.INT8, 1)]
        for v in range(depth.shape[1]):
            for u in range(depth.shape[0]):
                r = normals[0,u,v]
                g = normals[1,u,v]
                b = normals[2,u,v]
                z = depth[u,v] / scalingFactor
                # print(z)
                if z==0: continue
                x = (u - x0) * z / fx
                y = (v - y0) * z / fy
                points.append([x,y,z,r,g,b])
        cloud = PointCloud2(header=camera_info.header,
                            height=1,
                            width=len(points),
                            is_dense=False,
                            is_bigendian=False,
                            fields=fields,
                            point_step=15,
                            row_step=15 * len(points))
        
        self.pub_cloud.publish(cloud)
        print("cloud is published with "+str(cloud.width)+" points")

    def callback(self,camera_info,depth_msg):
        bridge = CvBridge()
        camera_info_K = np.array(camera_info.K)
        # print(camera_info_K)
        depth_image = bridge.imgmsg_to_cv2(depth_msg, 'passthrough')

        if len(depth_image.shape) < 3:
            print("Got 1 channel depth images, creating 3 channel depth images")
            combine_depth = np.empty((depth_image.shape[0],depth_image.shape[1], 3))
            combine_depth[:,:,0] = depth_image
            combine_depth[:,:,1] = depth_image
            combine_depth[:,:,2] = depth_image
            depth = combine_depth
        depth2 = np.moveaxis(cv2.resize(depth,(depth.shape[1],depth.shape[0])).astype(np.float32),-1,0)
        self.img = torch.from_numpy(depth2).float().unsqueeze(0)
        start = timeit.default_timer()
        z_fake = self.d2n(self.img.cuda())
        stop = timeit.default_timer()
        print('Predicting the image took ' + str(stop-start) + " seconds")
        zfv=z_fake*2-1
        z_fake_norm=zfv.pow(2).sum(dim=1).pow(0.5).unsqueeze(1)
        zfv=zfv/z_fake_norm
        z_fake=(zfv+1)/2
        # save_path=args.input_image_path[:-4]
        # save_image(z_fake[0], save_path +"_pred"+'.png')
        self.publish_pcd(camera_info,depth_image,z_fake[0])
    

def listener():    
    
    global d2n, args, img, rospy, pub_cloud
    estimator=Estimator()       
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows() 

if __name__ == '__main__':
    listener()
    