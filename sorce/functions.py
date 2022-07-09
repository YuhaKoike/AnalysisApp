import os
import glob
import csv
import math
import cv2
import numpy as np
import configparser

def ROIimg(inputimg, pt):
    return inputimg[pt[0][1]:pt[1][1], pt[0][0]:pt[1][0]]

def DSTimg(originimg, inputimg, pt):
    originimg[pt[0][1]:pt[1][1], pt[0][0]:pt[1][0]] = inputimg
    return originimg

def BinConv(inputimg, th):
    if len(inputimg.shape) == 2:
        img = inputimg
    else:
        img = cv2.cvtColor(inputimg, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, th, 255, cv2.THRESH_BINARY)
    img = cv2.bitwise_not(img)
    return img

def nearPoint(basepoint, points):
    result = np.empty([1,2])
    if len(points) == 0:
        return result
    result = points[0]
    stdval = -1
    for point in points:
        distance = math.sqrt((point[0]-basepoint[0])**2 + (point[1]-basepoint[1])**2)
        if stdval > distance or stdval == -1:
            result = point
            stdval = distance
    return result

def CalcGeledge(inputimg, th, roi):
    img_bin = BinConv(inputimg, th)
    kernel = np.ones((3,3), np.uint8)
    img_bin = cv2.dilate(img_bin, kernel, iterations=1)
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not len(contours) == 0:
        max_cnt = max(contours, key=lambda x: cv2.arcLength(x, True))
    for cnt in max_cnt:
        cnt[0] += roi[0]
    return max_cnt

def CalcBeadfeatures(inputimg, th, basepoints, roi):
    img_bin = BinConv(inputimg, th)
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = list(filter(lambda x: cv2.contourArea(x) >= 10, contours))
    medians = np.empty([0,2])
    if len(contours) == 0:
        print('No contours')
        return
    else:      
        for cnt in contours:
            mu = cv2.moments(cnt)
            x,y = int(mu['m10']/mu['m00']), int(mu['m01']/mu['m00'])
            medians = np.vstack([medians, (x,y)])

        points = np.empty([0,2])
        for basepoint in basepoints:
            point = nearPoint(basepoint-roi[0], medians)
            points = np.vstack([points, point])
    
    return points

def DetectBeadfeature(DATA, features_prev):
    CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    features, status, err = cv2.calcOpticalFlowPyrLK(DATA.pimg_crop, DATA.img_crop, \
                                                        features_prev.astype(np.float32), None, \
                                                        winSize = (5, 5), maxLevel = 3, \
                                                        criteria = CRITERIA, flags = 0)
    if features is None:
        print('Optical Flow Error')
        return
    i=0
    while i < len(features):
        if status[i] == 0:
            features = np.delete(features, i, 0)
            status = np.delete(status, i, 0)
            i -= 1
        i += 1

    return features

def CalcGelpoint(geledge, features, gelmedian):
    gelpoints = np.empty([0,2])
    for feature in features:
        min = -1
        gelpoint = np.empty(0)
        for point in geledge:
            a = np.linalg.norm(gelmedian-point[0])
            b = np.linalg.norm(point[0]-feature)
            dis = a+b
            if min == -1:
                min = a+b
            elif min > dis:
                min = dis
                gelpoint = point[0].copy()

        gelpoints = np.vstack([gelpoints,(gelpoint)])
    
    return gelpoints

def SaveDatalist(window, values, cap, ret_scale, filename, fps, width, height, bar_max, layoutlist, startframe, contrast, brightness):
    with open(filename + '.csv', 'w', newline='') as csv_file:
        csvwriter = csv.writer(csv_file)                   
        videowriter = cv2.VideoWriter(filename+'_output.avi', 0, fps, (width, height))

        for i in range(bar_max + 1):
            ret, frame = cap.read()
            if ret:
                saveframe = frame.copy()
                frame = frame * contrast + brightness
                frame = np.clip(frame, 0, 255).astype(np.uint8)
                
                for l in layoutlist:
                    NAME, DATA = l
                    DATA.conv = (DATA.roi/ret_scale).astype(np.int32)
                    DATA.img_crop = ROIimg(frame, DATA.conv)
                    if values['Posi'+NAME]:
                        DATA.img_crop = cv2.bitwise_not(DATA.img_crop)
                    if NAME == 'gel':
                        geledge = CalcGeledge(DATA.img_crop, DATA.th, DATA.conv)
                        if i == 0:
                            mu = cv2.moments(geledge)
                            gelmedian = np.array([mu['m10']/mu['m00'], mu['m01']/mu['m00']])
                    if NAME == 'bead':
                        if i == 0:
                            features = CalcBeadfeatures(DATA.img_crop, DATA.th, DATA.basepoints/ret_scale, DATA.conv)
                        else:
                            features = DetectBeadfeature(DATA, features_prev)

                        features_prev = features.copy()
                        DATA.pimg_crop = DATA.img_crop
                
                for feature in features:
                    feature += DATA.conv[0]
                    saveframe = cv2.circle(saveframe, tuple(feature.astype(np.int32)), 2, (0, 0, 255), 2)

                gelpoints = CalcGelpoint(geledge, features, gelmedian)
                saveframe = cv2.circle(saveframe, tuple(gelmedian.astype(np.int32)), 2, (255, 0, 0), 2)

                for gelpoint in gelpoints:
                    saveframe = cv2.circle(saveframe, tuple(gelpoint.astype(np.int32)), 2, (0, 255, 0), 2)
                
                if i == 0:
                    data = [['fps', fps], ['width', frame.shape[0]], ['height', frame.shape[1]], ['startframe', startframe], ['endframe', bar_max-startframe], ['median'] + gelmedian.tolist(), []]
                    csvwriter.writerows(data)
                    data = ['',]
                    for j in range(len(features)):
                        data.extend(['gel'+str(j), '', 'bead'+str(j), ''])
                    csvwriter.writerow(data)
                    data = ['Frame', ]
                    for j in range(len(features)):
                        X = 'x'+str(j)
                        Y = 'y'+str(j)
                        data.extend([X,Y,X,Y])
                    csvwriter.writerow(data)

                data = [startframe + i, ]
                for j in range(len(features)):
                    data += gelpoints[j].tolist() + features[j].tolist()
                csvwriter.writerow(data)
                videowriter.write(saveframe)
                window['-PROG-'].update(i+1)
                window['-PROGNOW-'].update(str(int(100*i/bar_max))+'%')

            else:
                window['-PROG-'].update(bar_max)
                window['-PROGNOW-'].update('100%')

    videowriter.release()

def InputIni(window, CustomButtons, layoutlist):
    CustomButtons[2].click()
    event, values = window.read()
    path = values['inputini']
    config = configparser.RawConfigParser()
    config.read(path)
    
    for l in layoutlist:
        NAME, DATA = l
        DATA.roi = np.empty([0,2])
        for i in range(1,3):
            x = float(config.get('roi'+NAME,'x'+str(i)))
            y = float(config.get('roi'+NAME,'y'+str(i)))
            DATA.roi = np.vstack([DATA.roi, (x,y)])
            window['x'+str(i)+NAME].update(x)
            window['y'+str(i)+NAME].update(y)

        th = int(config.get('threshold'+NAME, 'value'))
        DATA.th = th
        window['thtext'+NAME].update(th)
        window['thinput'+NAME].update(th)

    #frame range
    window['startframe'].update(int(config.get('frame range', 'start')))
    window['endframe'].update(int(config.get('frame range', 'end')))


def OutputIni(window, CustomButtons, layoutlist):
    CustomButtons[3].click()
    event, values = window.read()
    path = values['outputini']
    config = configparser.RawConfigParser()

    for l in layoutlist:
        NAME, DATA = l
        if len(DATA.roi) == 2:
            section = 'roi'+NAME
            config.add_section(section)
            for i in range(1,3):
                config.set(section, 'x'+str(i), DATA.roi[i-1][0])
                config.set(section, 'y'+str(i), DATA.roi[i-1][1])

        section = 'threshold'+NAME
        config.add_section(section)
        config.set(section, 'value', DATA.th)

    section = 'frame range'
    config.add_section(section)
    config.set(section, 'start', values['startframe'])
    config.set(section, 'end', values['endframe'])
    
    with open(path, 'w') as file:
        config.write(file)

def videocompress(val, path, popup):
    conv2gray = False
    compress = False
    if 'conv2gray' in val:
        conv2gray = True
        print('Convert to gray')
    if 'compress' in val:
        compress = True
        print('Compress')

    for file in glob.glob(path + '/*.avi'):
        p = os.path.splitext(file)
        outpath = p[0] + '_edit.avi'
        video = cv2.VideoCapture(file)
        size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        popup.set_prog(length)

        fps = int(video.get(cv2.CAP_PROP_FPS))
        fmt = 0
        if compress:
            fmt = cv2.VideoWriter_fourcc('P','I','M','1')

        writer = cv2.VideoWriter(outpath, fmt, fps, size, not conv2gray)
        for i in range(length):
            ret, frame = video.read()
            if ret:
                if conv2gray:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                writer.write(frame)
            else:
                writer.release()
                video.release
                break
            popup.update_prog(i)
        
    print('End convert')

class ExWindow:
    def __init__(self):
        pass
    def get_value(self):
        pass
    def close(self):
        pass

class progbar:
    def __init__(self):
        pass
