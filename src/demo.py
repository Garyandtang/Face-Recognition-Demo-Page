import logging
import nerual_net
import pandas as pd
import cv2
import numpy as np
from matlab_cp2tform import get_similarity_transform_for_cv2
import torch
import csv
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from sklearn import preprocessing
import lfw_verification_for_validate
import lfw_verification
import common_args
import os
import torchvision.models as models
from flask import Flask, flash, jsonify, redirect, request,render_template,url_for,Response
from werkzeug.utils import secure_filename
face_folder = os.path.join('static', 'face')
save_folder = os.path.join('static', 'save')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

model_root = '/home/tangjiawei/project/fyp/saved/30000_net_backbone.pth'
_lfw_root = '/home/tangjiawei/project/dataset/lfw/'
_lfw_landmarks = '/home/tangjiawei/project/dataset/LFW.csv'
_lfw_pairs = '/home/tangjiawei/project/dataset/lfw_pairs.txt'

# initialize flask application
app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = face_folder
app.config['SAVE_FOLDER'] = save_folder
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


def alignment(src_img, src_pts, size=None):
    ref_pts = [[30.2946, 51.6963], [65.5318, 51.5014],
               [48.0252, 71.7366], [33.5493, 92.3655],
               [62.7299, 92.2041]]
    if size is not None:
        ref_pts = np.array(ref_pts)
        ref_pts[:,0] = ref_pts[:,0] * size/96
        ref_pts[:,1] = ref_pts[:,1] * size/96
        crop_size = (int(size), int(112/(96/size)))
    else:
        crop_size = (96, 112)
    src_pts = np.array(src_pts).reshape(5, 2)
    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)
    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size, flags=cv2.INTER_CUBIC)
    if size is not None:
        face_img = cv2.resize(face_img, dsize=(96, 112), interpolation=cv2.INTER_CUBIC)
    return face_img

def get_alignedface(file_path, name):
    df = pd.read_csv(_lfw_landmarks, delimiter=",", header=None)
    numpyMatrix = df.values
    landmarks = numpyMatrix[:, 1:]
    p = name.rsplit('_', 1)
    name = p[0] + '/' + p[0] + '_' + p[1]
    img = alignment(cv2.imread(file_path), landmarks[df.loc[df[0] == name].index.values[0]])
    return img

def face_ToTensor(img):
    return (ToTensor()(img) - 0.5) * 2

@app.route("/")
def gui():
    return render_template('demo.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['POST'])
def handle_data():
    args = common_args.get_args()
    device = torch.device(1)

    # Receive data from front-end
    print("received")
    if 'file1' not in request.files:
        # flash('No file part')
        # print("test point 1")
        # return redirect(request.url)
        csvFile = open("store_name.csv", "r")
        reader = csv.reader(csvFile)
        for item in reader:
            print(item)
        filename1 = item[0]
        filename2 = item[1]
        csvFile.close()
    else:
        file1 = request.files['file1']
        file2 = request.files['file2']
        if file1.filename == '' or file2.filename == '':
            flash('No selected file')
            print("test point 2")
            return redirect(request.url)
        # saved_file1 = file1.filename
        # saved_file2 = file2.filename
        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)
        csvFile = open("store_name.csv", "w")
        writer = csv.writer(csvFile)
        writer.writerow([filename1, filename2])
        csvFile.close()
    print(filename1)
    print(filename2)
    method = int(request.values.get("method"))
    resolution1 = int(request.values.get("resolution1"))
    resolution2 = int(request.values.get("resolution2"))
    # Deal with received data, including images, model and resolution
    p = filename1.rsplit('_', 1)
    file_path1 = p[0] + '/' + p[0] + '_' + p[1]
    p = filename2.rsplit('_', 1)
    file_path2 = p[0] + '/' + p[0] + '_' + p[1]
    file_path1 = os.path.join(app.config['UPLOAD_FOLDER'], file_path1)
    file_path2 = os.path.join(app.config['UPLOAD_FOLDER'], file_path2)
    print(file_path1)
    print(filename1)
    resized_face1 = cv2.imread(file_path1)
    resized_face2 = cv2.imread(file_path2)
    print("check")
    width1, height1, channel1 = resized_face1.shape
    width2, height2, channel2 = resized_face2.shape
    print("check")
    # cv2.resize(face, (int(96 / self.N), int(112 / self.N)))
    resized_face1 = cv2.resize(resized_face1, (int(width1 / resolution1), int(height1 / resolution1)))
    print("check")
    resized_face1 = cv2.resize(resized_face1, (width1, height1))
    print("check")
    resized_face2 = cv2.resize(resized_face2, (int(width2 / resolution2), int(height2 / resolution2)))
    resized_face2 = cv2.resize(resized_face2, (width2, height2))
    resized_face_path1 = os.path.join(app.config['UPLOAD_FOLDER'], 'resized_face1.jpg')
    resized_face_path2 = os.path.join(app.config['UPLOAD_FOLDER'], 'resized_face2.jpg')
    cv2.imwrite(resized_face_path1, resized_face1)
    cv2.imwrite(resized_face_path2, resized_face2)
    aligned_face1 = get_alignedface(file_path1, filename1)
    aligned_face2 = get_alignedface(file_path2, filename2)
    aligned_face1 = cv2.resize(aligned_face1, (int(96 / resolution1), int(112 / resolution1)))
    aligned_face2 = cv2.resize(aligned_face2, (int(96 / resolution2), int(112 / resolution2)))
    aligned_face1 = cv2.resize(aligned_face1, (96, 112))
    aligned_face2 = cv2.resize(aligned_face2, (96, 112))
    aligned_face_path1 = os.path.join(app.config['UPLOAD_FOLDER'], 'aligned_face1.jpg')
    aligned_face_path2 = os.path.join(app.config['UPLOAD_FOLDER'], 'aligned_face2.jpg')
    cv2.imwrite(aligned_face_path1, aligned_face1)
    cv2.imwrite(aligned_face_path2, aligned_face2)

    # setup model
    if method == 1:
        model_root1 = '/home/tangjiawei/project/fyp/saved/30000_net_backbone.pth'
        model_root2 = '/home/tangjiawei/PycharmProjects/FYP_Face_Verification/saved/1000_lr_spherenet_df7.pth'
        net1 = nerual_net.spherenet(20, args.feature_dim, args.use_pool, args.use_dropout)
        net1.load_state_dict(torch.load(model_root1))
        net1.to(device)
        net1.eval()
        net2 = nerual_net.spherenet(20, args.feature_dim, args.use_pool, args.use_dropout)
        net2.load_state_dict(torch.load(model_root2))
        net2.to(device)
        net2.eval()
        feature1 = net1(face_ToTensor(aligned_face1).to(device).view([1, 3, 112, 96]))
        feature2 = net1(face_ToTensor(aligned_face2).to(device).view([1, 3, 112, 96]))
        feature3 = net2(face_ToTensor(aligned_face1).to(device).view([1, 3, 112, 96]))
        feature4 = net2(face_ToTensor(aligned_face2).to(device).view([1, 3, 112, 96]))
        model = "Resnet"
        thd_res_hr_diff_group = [0, 0.286, 0, 0, 0.25, 0, 0.152, 0.13, 0, 0, 0.06, 0, 0.0226]
        thd_res_hr_same_group = [0, 0.286, 0, 0, 0.225, 0, 0.203, 0.178, 0, 0, 0.268, 0, 0.0226]
        thd_res_lr_diff_group = [0, 0.343, 0, 0, 0.307, 0, 0.257, 0.2256, 0, 0, 0.1414, 0, 0.071]
        thd_res_lr_same_group = [0, 0.344, 0, 0, 0.297, 0, 0.269, 0.235, 0, 0, 0.24, 0, 0.295]
        if resolution2 == 1:
            thd_hr = thd_res_hr_diff_group[resolution1]
            thd_lr = thd_res_lr_diff_group[resolution1]
        elif resolution1 == 1:
            thd_hr = thd_res_hr_diff_group[resolution2]
            thd_lr = thd_res_lr_diff_group[resolution2]
        else:
            thd_hr = thd_res_hr_same_group[resolution1]
            thd_lr = thd_res_lr_same_group[resolution1]

    if method == 2:
        model_root1 = '/home/tangjiawei/project/fyp/saved/ALEX_NET_HR_11_April.pkl'
        model_root2 = '/home/tangjiawei/PycharmProjects/RadimoicDeepFeatureExtraction/model/1000_lr_alexnet_df9.pth'
        net1 = models.alexnet(pretrained=True)
        net1.classifier[6] = torch.nn.Linear(4096, 10559)
        net1.load_state_dict(torch.load(model_root1))
        net1.to(device)
        net1.eval()
        net2 = models.alexnet(pretrained=True)
        net2.classifier[6] = torch.nn.Linear(4096, 10559)
        net2.load_state_dict(torch.load(model_root2))
        net2.to(device)
        aligned_face1 = cv2.resize(aligned_face1, (224, 224))
        aligned_face2 = cv2.resize(aligned_face2, (224, 224))
        net2.eval()
        feature1 = net1(face_ToTensor(aligned_face1).to(device).view([1, 3, 224, 224]))
        feature2 = net1(face_ToTensor(aligned_face2).to(device).view([1, 3, 224, 224]))
        feature3 = net2(face_ToTensor(aligned_face1).to(device).view([1, 3, 224, 224]))
        feature4 = net2(face_ToTensor(aligned_face2).to(device).view([1, 3, 224, 224]))
        model = "Alexnet"
        thd_res_hr_diff_group = [0, 0.535, 0, 0, 0.472, 0, 0.4, 0.338, 0, 0, 0.274, 0, 0.226]
        thd_res_hr_same_group = [0, 0.55, 0, 0, 0.485, 0, 0.439, 0.49, 0, 0, 0.58, 0, 0.59]
        thd_res_lr_diff_group = [0, 0.682, 0, 0, 0.61, 0, 0.543, 0.47, 0, 0, 0.40, 0, 0.522]
        thd_res_lr_same_group = [0, 0.687, 0, 0, 0.6, 0, 0.557, 0.512, 0, 0, 0.45, 0, 0.507]
        if resolution2 == 1:
            thd_hr = thd_res_hr_diff_group[resolution1]
            thd_lr = thd_res_lr_diff_group[resolution1]
        elif resolution1 == 1:
            thd_hr = thd_res_hr_diff_group[resolution2]
            thd_lr = thd_res_lr_diff_group[resolution2]
        else:
            thd_hr = thd_res_hr_same_group[resolution1]
            thd_lr = thd_res_lr_same_group[resolution1]
    # get features from received images, comparing two images --> same or different
    # feature1 = net1(face_ToTensor(aligned_face1).to(device).view([1, 3, 112, 96]))
    # feature2 = net1(face_ToTensor(aligned_face2).to(device).view([1, 3, 112, 96]))
    print(feature1.shape)
    print(feature2.shape)
    score1 = torch.nn.CosineSimilarity()(feature1, feature2)
    score1 = score1.cpu().detach().numpy().reshape(-1, 1)
    score1 = score1.item()
    score2 = torch.nn.CosineSimilarity()(feature3, feature4)
    score2 = score2.cpu().detach().numpy().reshape(-1, 1)
    score2 = score2.item()
    print(score1)
    print(score2)
    print(thd_lr)
    print(thd_hr)
    if score1 >= thd_hr:
        text_to_html1 = model + " verification result: Same Person"
        confindence_text1 = "Score is {:.4f}".format(score1)
    else:
        text_to_html1 = model + " verification result: Different Persons"
        confindence_text1 = "Score is {:.4f}".format(score1)
    if score2 >= thd_lr:
        text_to_html2 = "LR-" + model + " verification result: Same Person"
        confindence_text2 = "Score is {:.4f}".format(score2)
    else:
        text_to_html2 = "LR-" + model + " verification result: Different Persons"
        confindence_text2 = "Score is {:.4f}".format(score2)



    return render_template('demo.html', input_face1=resized_face_path1, input_face2=resized_face_path2, aligned_face1=aligned_face_path1,\
                           aligned_face2=aligned_face_path2,Result_test1=text_to_html1, Confidence1=confindence_text1,\
                           Result_test2=text_to_html2, Confidence2=confindence_text2)
    # if file and allowed_file(file.filename):
    #     filename = secure_filename(file.filename)
    #     file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    #     aligned_face = get_alignedface(file_path, filename)
    #     aligned_face_path = os.path.join(app.config['UPLOAD_FOLDER'], 'aligned_face.jpg')
    #     cv2.imwrite(aligned_face_path, aligned_face)

    # file = request.files['file1']
    # filename = secure_filename(file.filename)
    # file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    # print(file)
    # print(file_path)
    # return render_template('demo.html', input_face="static/face/Zinedine_Zidane_0006.jpg", input_face_path=filename)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8003)

    print('Done')
    # downsampling_factor = 7
    # args = common_args.get_args()
    # device = torch.device(1)
    # print(device)
    #
    # df = pd.read_csv(_lfw_landmarks, delimiter=",", header=None)
    # numpyMatrix = df.values
    # landmarks = numpyMatrix[:, 1:]
    # with open(_lfw_pairs) as f:
    #     pairs_lines = f.readlines()[1:]
    # p = pairs_lines[0].replace('\n', '').split('\t')
    # if 3 == len(p):
    #     sameflag = np.int32(1).reshape(1)
    #     name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
    #     name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
    # if 4 == len(p):
    #     sameflag = np.int32(0).reshape(1)
    #     name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
    #     name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
    # img1 = alignment(cv2.imread(_lfw_root + name1),
    #                  landmarks[df.loc[df[0] == name1].index.values[0]])
    # img2 = alignment(cv2.imread(_lfw_root + name2),
    #                  landmarks[df.loc[df[0] == name2].index.values[0]])
    #
    #
    # spherenet_64 = nerual_net.spherenet(20, args.feature_dim, args.use_pool, args.use_dropout)
    # spherenet_64.load_state_dict(torch.load('/home/tangjiawei/PycharmProjects/FYP_Face_Verification/saved/lr_spherenet_hr10_lr20_model/2400_lr_spherenet2064_df7.pth'))
    # spherenet_64.to(device)
    # spherenet_64.eval()
    #
    # feature1 = spherenet_64(face_ToTensor(img1).to(device).view([1,3,112,96]))
    # print(feature1)
    # # input_feature - spherenet_64()
