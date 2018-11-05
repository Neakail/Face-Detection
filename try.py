# -*- coding:utf-8 -*-
from PIL import Image
import glob
import pickle
from sklearn.svm import SVC,LinearSVC
from skimage import feature as ft
from pylab import *
from sklearn.decomposition import PCA
def ImageToMatrix(filename):
    # 读取图片
    im = Image.open(filename)
    width,height = im.size
    im = im.convert("L")
    data = im.getdata()
    data = np.matrix(data,dtype='float')/255.0
    new_data = np.reshape(data,(width,height))
    return new_data

def MatrixToImage(data):
    data = data*255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im

def JfzBlogImgThumb(im):
    # im = Image.open(ImgName)
    print('格式',im.format, '，分辨率',im.size, '，色彩',im.mode)
    im.show()
    im.thumbnail((19, 19))
    print('格式', im.format, '，分辨率', im.size, '，色彩', im.mode)
    im.show()


def cut(img,x1, y1, x2, y2):
    box = (x1, y1, x2, y2)
    region = img.crop(box)
    return region

pca = PCA(n_components=1,copy=True)
def transmit(filename):
    im = array(Image.open(filename).convert('L'), 'f')
    newData = pca.fit_transform(im)
    a = []
    for i in newData:
        a.append(i[0])
    print a
    return a

def hog(filename):
    image = array(Image.open(filename).convert('L'), 'f')
    ori = 9
    ppc = [16, 16]
    cpb = [1, 1]
    features = ft.hog(image,  # input image
                  orientations=ori,  # number of bins
                  pixels_per_cell=ppc,  # pixel per cell
                  cells_per_block=cpb,  # cells per block
                  block_norm='L1',  # block norm : str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}
                  transform_sqrt=True,  # power law compression (also known as gamma correction)
                  feature_vector=True,  # flatten the final vectors
                  visualise=False)  # return HOG map
    # print features
    return features

def hog_(image):
    ori = 9
    ppc = [16, 16]
    cpb = [1, 1]
    features = ft.hog(image,  # input image
                  orientations=ori,  # number of bins
                  pixels_per_cell=ppc,  # pixel per cell
                  cells_per_block=cpb,  # cells per block
                  block_norm='L1',  # block norm : str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}
                  transform_sqrt=True,  # power law compression (also known as gamma correction)
                  feature_vector=True,  # flatten the final vectors
                  visualise=False)  # return HOG map
    # print features
    return features


def transmit1(im):
    newData = pca.fit_transform(im)
    a = []
    for i in newData:
        a.append(i[0])
    # print a
    return a

def train():
    x = []
    y = []
    negative_number = 0
    filenames = glob.glob('/media/linnankai/Elements/negative/*')
    for filename in filenames:
        if negative_number < 10000:
            x.append(hog(filename))
            y.append(0)
            negative_number += 1
            print negative_number
        else:
            break
    positive_number = 0
    filenames = glob.glob('/media/linnankai/Elements/positive/*')
    for filename in filenames:
        if positive_number < 10000:
            x.append(hog(filename))
            y.append(1)
            positive_number += 1
            print positive_number
        else:
            break

    X = np.array(x)
    y = np.array(y)
    clf = SVC(probability=True,kernel="linear")
    # cls = LinearSVC()
    clf.fit(X, y)

    with open('SVC1.pickle', 'w') as f:
        pickle.dump(clf, f)


def transmit224(region):
    region = region.resize((224,224))
    return region


def draw_frame(filename,face):
    color = (0, 255, 0)
    import cv2
    region = cv2.imread(filename)
    for each in face:
        cv2.rectangle(region, (each[0], each[1]), (each[2],each[3]), color, 2)
    cv2.imwrite('1/' + filename,region)

def move(frame_size,filename):

    face = []
    with open('SVC1.pickle') as f:
        clf = pickle.load(f)
    region = Image.open(filename).convert('L')
    region_array = array(region,'f')
    height = len(region_array)
    width = len(region_array[0])

    b_ = []
    for i in range(0,height-frame_size,20):
        for j in range(0,width-frame_size,20):
            a = cut(region,i,j,i+frame_size,j+frame_size)
            a = transmit224(a)
            a = array(a,'f')
            b = hog_(a)
            b_.append(b)

    gailvs = clf.predict_proba(b_)

    for gailv in gailvs:
        gailv = gailv[0][1]
        # print gailv
        if gailv > 0.8:
            # print gailv
            temp = {}
            temp['coordinate'] = [i,j,i+frame_size,j+frame_size]
            temp['tag'] = 1
            temp['gailv'] = gailv
            temp['size'] = frame_size
            # temp = [i,j,i+frame_size,j+frame_size]
            face.append(temp)
    return face

def NMS(pic1,pic2):
    xx1 = max(pic1['coordinate'][0],pic2['coordinate'][0])
    yy1 = max(pic1['coordinate'][1], pic2['coordinate'][1])
    xx2 = min(pic1['coordinate'][2],pic2['coordinate'][2])
    yy2 = min(pic1['coordinate'][3], pic2['coordinate'][3])
    w = xx2 - xx1
    h = yy2 - yy1
    o = 0
    if w > 0 and h >0:
        o = 1.0 * w * h / min(pic1['size']*pic1['size'],pic2['size']*pic2['size'])
        print o
    return o


def run(filename):
    face = []
    face_ = []
    frame_sizes = [60, 80, 100, 120, 140]
    for i in frame_sizes:
        a = move(i, filename)
        for j in a:
            face.append(j)
    print len(face)
    for i in face:
        if i['tag'] == 0:
            continue
        else:
            for j in face:
                if i == j or j['tag'] == 0:
                    continue
                else:
                    score = NMS(i, j)
                    if score > 0.2:
                        if i['gailv'] > j['gailv']:
                            j['tag'] = 0
                            # print '----------------------'
                        else:
                            i['tag'] = 0
                            # print '-----------------------'
                            break

    for i in face:
        if i['tag'] == 1:
            face_.append(i['coordinate'])

    draw_frame(filename, face_)


if __name__ == '__main__':
    # face = []
    # filename = '5.jpg'
    # frame_sizes = [60,80,100,120,140]
    # for i in frame_sizes:
    #     a = run(i,filename)
    #     for j in a:
    #         face.append(j)
    #draw_frame(filename, face)

    run()