import logging
import nerual_net
import torch
import lfw_verification_for_validate
import lfw_verification
import common_args
logging.basicConfig(filename='FYP_Resnet.log', level=logging.INFO)
model_root = '/home/tangjiawei/project/fyp/saved/30000_net_backbone.pth'
if __name__ == '__main__':
    downsampling_factor = 7
    args = common_args.get_args()
    device = torch.device(1)
    print(device)

    # # Setup network Spherenet20_lR
    # num_layers = 20
    # lr_spherenet_test = nerual_net.spherenet(num_layers, args.feature_dim, args.use_pool, args.use_dropout)
    # lr_spherenet_test.load_state_dict(
    #     torch.load('/home/tangjiawei/PycharmProjects/FYP_Face_Verification/saved/2000_lr_spherenet_df9.pth'))
    # lr_spherenet_test.to(device)
    #
    # for i in range(16):
    #     lfw_verification_for_validate.run(lr_spherenet_test, feature_dim=512, device=device, N=(i + 1))

    spherenet_64 = nerual_net.spherenet(20, args.feature_dim, args.use_pool, args.use_dropout)
    spherenet_64.load_state_dict(torch.load(model_root))
    spherenet_64.to(device)
    logging.info('test sphereface backbone img1_lr img2_hr')
    for i in range(16):
        lfw_verification.run(spherenet_64,feature_dim=512,device=device, N=(i+1))

# import logging
# import nerual_net
# import pandas as pd
# import cv2
# import numpy as np
# from matlab_cp2tform import get_similarity_transform_for_cv2
# import torch
# from torchvision.transforms import ToTensor
# from torch.utils.data import DataLoader
# from sklearn import preprocessing
# import lfw_verification_for_validate
# import lfw_verification
# import common_args

# logging.basicConfig(filename='FYP_Resnet.log', level=logging.INFO)
# model_root = '/home/tangjiawei/project/fyp/saved/30000_net_backbone.pth'
# _lfw_root = '/home/tangjiawei/project/dataset/lfw/'
# _lfw_landmarks = '/home/tangjiawei/project/dataset/LFW.csv'
# _lfw_pairs = '/home/tangjiawei/project/dataset/lfw_pairs.txt'
#
# def alignment(src_img, src_pts, size=None):
#     ref_pts = [[30.2946, 51.6963], [65.5318, 51.5014],
#                [48.0252, 71.7366], [33.5493, 92.3655],
#                [62.7299, 92.2041]]
#     if size is not None:
#         ref_pts = np.array(ref_pts)
#         ref_pts[:,0] = ref_pts[:,0] * size/96
#         ref_pts[:,1] = ref_pts[:,1] * size/96
#         crop_size = (int(size), int(112/(96/size)))
#     else:
#         crop_size = (96, 112)
#     src_pts = np.array(src_pts).reshape(5, 2)
#     s = np.array(src_pts).astype(np.float32)
#     r = np.array(ref_pts).astype(np.float32)
#     tfm = get_similarity_transform_for_cv2(s, r)
#     face_img = cv2.warpAffine(src_img, tfm, crop_size, flags=cv2.INTER_CUBIC)
#     if size is not None:
#         face_img = cv2.resize(face_img, dsize=(96, 112), interpolation=cv2.INTER_CUBIC)
#     return face_img
# def face_ToTensor(img):
#     return (ToTensor()(img) - 0.5) * 2
#
# if __name__ == '__main__':
#     downsampling_factor = 7
#     args = common_args.get_args()
#     device = torch.device(1)
#     print(device)
#
#     df = pd.read_csv(_lfw_landmarks, delimiter=",", header=None)
#     numpyMatrix = df.values
#     landmarks = numpyMatrix[:, 1:]
#     with open(_lfw_pairs) as f:
#         pairs_lines = f.readlines()[1:]
#     p = pairs_lines[0].replace('\n', '').split('\t')
#     if 3 == len(p):
#         sameflag = np.int32(1).reshape(1)
#         name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
#         name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
#     if 4 == len(p):
#         sameflag = np.int32(0).reshape(1)
#         name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
#         name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
#     img1 = alignment(cv2.imread(_lfw_root + name1),
#                      landmarks[df.loc[df[0] == name1].index.values[0]])
#     img2 = alignment(cv2.imread(_lfw_root + name2),
#                      landmarks[df.loc[df[0] == name2].index.values[0]])
#
#
#     spherenet_64 = nerual_net.spherenet(20, args.feature_dim, args.use_pool, args.use_dropout)
#     spherenet_64.load_state_dict(torch.load('/home/tangjiawei/PycharmProjects/FYP_Face_Verification/saved/lr_spherenet_hr10_lr20_model/2400_lr_spherenet2064_df7.pth'))
#     spherenet_64.to(device)
#     spherenet_64.eval()
#
#     feature1 = spherenet_64(face_ToTensor(img1).to(device).view([1,3,112,96]))
#     print(feature1)
#     # input_feature - spherenet_64()
