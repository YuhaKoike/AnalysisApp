infomation = 'This software is created by YK in 29/01/2022'

import os
import cv2
import numpy as np
import PySimpleGUI as sg

import functions as myfn
    
def resize(src, size):
    if len(src.shape) == 3:
        h, w, c = src.shape
    else:
        h, w = src.shape

    # aspect rate
    scale_w = size[0] / w
    scale_h = size[1] / h
    ret_scale = 1.0

    if scale_w < 1.0 or scale_h < 1.0:
        if scale_w < scale_h:
            resize_img = cv2.resize(src, dsize=None, fx=scale_w, fy=scale_w, interpolation = cv2.INTER_AREA)
            ret_scale = scale_w
        else:
            resize_img = cv2.resize(src, dsize=None, fx=scale_h, fy=scale_h, interpolation = cv2.INTER_AREA)
            ret_scale = scale_h
    else:
        resize_img = src
    
    return resize_img, ret_scale

def showimage(img, window):
    imgbytes = cv2.imencode('.png', img)[1].tobytes()
    window['image'].update(data=imgbytes)

def LeftPress(widget):
    return widget.user_bind_event.x, widget.user_bind_event.y

class DataInfo():
    def __init__(self, ):
        self.roi = np.empty([0,2])
        self.temppt = np.zeros(2)
        self.scale = 1.0
        self.status = False
        self.conv = np.empty([2,2])
        self.img_crop = np.empty(0)
        self.pimg_crop = np.empty(0)
        self.saveinfo = {}
        self.th = 100
        self.basepoints = np.empty([0,2])

def MainMenu():
    result = 0
    ret_scale = 1.0

    # ini settings

    # setting theme
    sg.theme('LightBlue')
    
    # create custom buttons
    CustomButtons = []
    def CustomButton(**args):
        obj = sg.Button(**args)
        CustomButtons.append(obj)
        return obj

    CustomButton(key='openfile', button_text='Open movie file', target=(sg.ThisRow, -1), file_types=(('Movie Files', '*.*'),),
                enable_events=True, button_type=sg.BUTTON_TYPE_BROWSE_FILES, visible=False)

    CustomButton(key='selectfolder', button_text='Select save folder', target=(sg.ThisRow, -1),
                enable_events=True, button_type=sg.BUTTON_TYPE_BROWSE_FOLDER, visible=False)

    CustomButton(key='inputini', button_text='Input init file', target=(sg.ThisRow, -1), file_types=(('Setting Files', '*.ini'),),
                enable_events=True, button_type=sg.BUTTON_TYPE_BROWSE_FILES, visible=False)

    CustomButton(key='outputini', button_text='Output init file', target=(sg.ThisRow, -1), file_types=(('Setting Files', '*.ini'),),
                enable_events=True, button_type=sg.BUTTON_TYPE_SAVEAS_FILE, visible=False,)

    # image frame
    movieinfo = '/0;(0.00 s)'
    movietitle = 'File name'

    image_layout = [[sg.Button('<', key='<', size=(2,1)), sg.Button('>', key='>', size=(2,1)), \
                    sg.Slider(range=(0,0), key='sliderframe', resolution=1,enable_events=True, default_value=0, orientation='h', disable_number_display=True, size=(30, 15)),\
                    sg.InputText('0', size=(8,1), key='nowframe', enable_events=True, justification='right'), \
                    sg.Text(movieinfo, key='movieinfo'), sg.Text(movietitle, key='movietitle')], 
                    [sg.Image(filename='', key='image', size=(1280, 720), enable_events=True)]
                    ]
    image_frame = sg.Frame('image', image_layout)

    # analysis frame
    startframe = 0
    endframe = 0
    bar_max = endframe - startframe
    cols = [(10, 1), (8, 1), (8, 1)]
    analysis_layout = [[sg.Text('Frame Range:', size=cols[0]), sg.InputText('0', size=(8,1), key='startframe', enable_events=True, justification='right'), \
                        sg.Text('-', justification='center'), sg.InputText('0', size=(8,1), key='endframe', enable_events=True, justification='right')],
                        [sg.Button('Start', key='start', size=(10, 1), enable_events=True)],
                        [sg.Text('0%',key='-PROGNOW-',justification='right', size=(4,1)), sg.ProgressBar(bar_max, orientation='h', size=(19,20), key='-PROG-')]
                        ]
    analysis_frame = sg.Frame('Analysis', analysis_layout)

    layoutlist = []
    def makelayout(name):
        data = DataInfo()
        layoutlist.append([name, data])
        layout = [[sg.Text('Roi:', size=(4, 1)), sg.Text('OFF', key='roitext'+name, size=(4, 1)), sg.Button('Setting', key='setroi'+name, size=(8, 1))],
                    [sg.Text('Top Left; ', size=cols[0]), sg.Text('x1', size=(2,1)), sg.InputText('0', size=(5,1), key='x1'+name,  justification='right'), \
                        sg.Text('y1', size=(2,1)), sg.InputText('0', size=(5,1), key='y1'+name, justification='right', enable_events=True)],
                    [sg.Text('Bottom Right; ', size=cols[0]), sg.Text('x2', size=(2,1)), sg.InputText('0', size=(5,1), key='x2'+name,  justification='right'), \
                        sg.Text('y2', size=(2,1)), sg.InputText('0', size=(5,1), key='y2'+name, justification='right', enable_events=True)],
                    [sg.Radio('Nega', key='Nega'+name, group_id='radio'+name, default=True, enable_events=True), sg.Radio('Posi', key='Posi'+name, group_id='radio'+name, enable_events=True)],
                    [sg.Checkbox('Threshold:', key = 'Threshold'+name, enable_events=True, size=(7,1)), sg.InputText('100', size=cols[1], key='thtext'+name, enable_events=True, justification='right'), \
                    sg.Slider(range=(0, 255), key='thinput'+name, enable_events=True, default_value=100, orientation='h', disable_number_display=True, size=(8, 15))]
                ]
        return layout



    # image processing frame
    #init
    contrast = 1.0
    brightness = 0
    gamma = 1.0
    imgp_arr = [{'name': 'Contrast', 'val': contrast, 'range': (0, 2.0), 'resolution': 0.1, 'default':1.0}, \
            {'name': 'Brightness', 'val': brightness, 'range': (-255, 255), 'resolution': 1, 'default':0}, \
            {'name': 'Gamma', 'val': gamma, 'range': (0, 2.0), 'resolution': 0.1, 'default':1.0}]

    imgp_layout = [[sg.Text(dic['name']+': ', size=(8,1)), sg.InputText(dic['val'], key='input'+dic['name'], size=(5,1), justification='right', enable_events=True),\
                    sg.Slider(range=dic['range'], resolution=dic['resolution'], key='slider'+dic['name'], enable_events=True, default_value=dic['default'],\
                    orientation='h', disable_number_display=True, size=(12,15))] for dic in imgp_arr]

    imgp_layout.append([sg.Button('Default', key='imgp_default', enable_events=True, size=(8,1))])

    imgp_frame = sg.Frame('Image process', imgp_layout)

    # gel frame
    gel_layout = makelayout('gel')
    gel_frame = sg.Frame('Gel', gel_layout)
    # bead frame
    bead_layout = makelayout('bead')
    bead_layout.append([sg.Checkbox('Select Beads', key='selectbeads', enable_events=True, size=(10,1))])
    bead_frame = sg.Frame('Bead', bead_layout)

    # Log
    log_layout = [[sg.Output(key='log', size=(30, 10))]]
    log_frame = sg.Frame('Log', log_layout)

    # Menu
    M_filelist = []
    for cnt in CustomButtons:
        M_filelist.append(cnt.ButtonText)
        M_filelist.append('---')
    M_filelist.append('Exit')

    # Edit
    M_editlist = ['Convert']
    class convWin(myfn.ExWindow):
        def __init__(self):
            button = sg.Button(button_text='Browse', key='selectfolder', target=(sg.ThisRow, -1),
            enable_events=True, button_type=sg.BUTTON_TYPE_BROWSE_FOLDER)
            self.layout = [[sg.Text('Folder Path'), sg.InputText(key='pathtext', disabled=True), button],\
                        [sg.Checkbox('Convert to Grayscale', key='conv2gray', enable_events=True, size=(20,1), default=False)],\
                        [sg.Checkbox('Compress', key='compress', enable_events=True, size=(20,1), default=False)],\
                        [sg.Text('File:'), sg.Text('0/0',key='-FILE-'), sg.Text('0%',key='-PROGNOW-',justification='right', size=(4,1)), sg.ProgressBar(bar_max, orientation='h', size=(40,20), key='-PROG-')],\
                        [sg.OK(), sg.Cancel()]]

            self.window = sg.Window(title='Convert Setting', layout=self.layout)
            self.excluded = set()
            self.names = ['conv2gray', 'compress']

        def get_value(self):
            path = None
            while True:
                event, values = self.window.read()
                if event in ('Exit', 'Quit', 'Cancel', None):
                    break
                elif event == 'OK':
                    for name in self.names:
                        if values[name]:
                            self.excluded.add(name)

                    if values['pathtext'] != '':
                        path = values['pathtext']
                    break
                elif event == 'selectfolder':
                    values['pathtext'] = values['selectfolder']
            return self.excluded, path

        def set_prog(self, max):
            self.max = max
            self.window['-PROG-'].update(0, max=max)

        def update_prog(self, i):
            self.window['-PROG-'].update(i)
            self.window['-PROGNOW-'].update(str(int(100*i/self.max))+'%')

        def close(self):
            self.window.close()

    menu = sg.Menubar([['File', M_filelist],\
                    ['Edit', M_editlist],\
                    ['Help', ['About software']]], key='menu')

    column = sg.Column([[analysis_frame], [imgp_frame], [gel_frame], [bead_frame],[log_frame]], vertical_alignment='top')
    layout = [  [menu],
                [image_frame, column],
                CustomButtons,    # unvisiible contents
            ]

    window = sg.Window('test', layout, return_keyboard_events=True, font=('Arial', 10))
    
    # add actions
    window.finalize()
    bindic = {'1': 'LeftPress', '2': 'WheelPress', '3': 'RightPress', 'Motion': 'MouseMove'}
    for key, eventname in bindic.items():
        window['image'].bind('<' + key + '>', eventname)

    originframe = None
    frame = None

    while(True):
        event, values = window.read()
        if event == sg.WIN_CLOSED or values['menu'] == 'Exit':
            result = -1
            break
        try:
            if event == 'About software':
                sg.popup(infomation)

            if event == 'Convert':
                popup = convWin()
                val, path = popup.get_value()
                if not path == None:
                    print(path)
                    myfn.videocompress(val, path, popup)
                popup.close()
                del popup

            if event == M_filelist[0]:
                CustomButtons[0].click()
                event, values = window.read()
                filepath = values['openfile']

                if os.path.splitext(filepath)[1] == '.avi' or '.mp4':
                    cap = cv2.VideoCapture(filepath)
                    ret, originframe = cap.read()
                    frame = originframe.copy()
                    if ret:
                        nowframe = int(values['nowframe'])
                        allframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        movieinfo = '/' + str(allframe) + ';(' + str(0 / fps) + ' s)'
                        window['sliderframe'].update(range=(0, allframe))
                        movietitle = os.path.basename(filepath)
                        window['movieinfo'].update(movieinfo)
                        window['movietitle'].update(movietitle)
                else:
                    frame = cv2.imread(filepath)

            if event == M_filelist[2]:
                CustomButtons[1].click()
                event, values = window.read()

            if event == M_filelist[4]:
                myfn.InputIni(window, CustomButtons, layoutlist)
                
            if event == M_filelist[6]:
                myfn.OutputIni(window, CustomButtons, layoutlist)

            if event == 'imageLeftPress':
                if values['selectbeads']:
                    DATA = layoutlist[1][1]
                    x,y = LeftPress(window['image'])
                    DATA.basepoints = np.vstack([DATA.basepoints, (x,y)])
                else:
                    for l in layoutlist:
                        NAME,DATA=l
                        if window['roitext' + NAME].get() == 'ON':
                            if len(DATA.roi) < 2:
                                DATA.roi = np.vstack([DATA.roi, LeftPress(window['image'])])
                                for i, xy in enumerate(DATA.roi.astype(np.int32),1):
                                    window['x'+str(i)+NAME].update(xy[0])
                                    window['y'+str(i)+NAME].update(xy[1])
            
            if event == 'imageRightPress':
                if values['selectbeads']:
                    DATA = layoutlist[1][1]
                    x,y = LeftPress(window['image'])
                    index = np.where(DATA.basepoints == myfn.nearPoint([x,y], DATA.basepoints))[0][0]
                    DATA.basepoints = np.delete(DATA.basepoints, index, 0)
                else:
                    for l in layoutlist:
                        NAME,DATA=l
                        if window['roitext'+NAME].get() == 'ON':
                            DATA.roi = np.empty([0,2])
                            DATA.temppt = np.zeros(2)
                            for i in range(1,3):
                                window['x'+str(i)+NAME].update(0)
                                window['y'+str(i)+NAME].update(0)

            if event == 'imageMouseMove':
                for l in layoutlist:
                    NAME,DATA=l
                    if window['roitext' + NAME].get() == 'ON':
                        if len(DATA.roi) == 1:
                            x,y = LeftPress(window['image'])
                            DATA.temppt = np.array([x,y])
                            window['x2'+NAME].update(x)
                            window['y2'+NAME].update(y)

            if event == '<':
                try:
                    nowframe = int(values['nowframe'])
                    nowframe -= 1
                    if nowframe < 0:
                        nowframe = 0
                    cap.set(cv2.CAP_PROP_POS_FRAMES, nowframe)
                    ret, frame = cap.read()
                    if ret:
                        window['nowframe'].update(nowframe)
                        movieinfo = '/' + str(allframe) + ';(' + str(nowframe / fps) + ' s)'
                        window['movieinfo'].update(movieinfo)
                except:
                    print('err')
                    continue

            if event == '>':
                try:
                    nowframe = int(values['nowframe'])
                    nowframe += 1
                    if nowframe > allframe:
                        nowframe = allframe
                    cap.set(cv2.CAP_PROP_POS_FRAMES, nowframe)
                    ret, frame = cap.read()
                    if ret:
                        window['nowframe'].update(nowframe)
                        movieinfo = '/' + str(allframe) + ';(' + str(nowframe / fps) + ' s)'
                        window['movieinfo'].update(movieinfo)
                except:
                    print('err')
                    continue

            if event == 'sliderframe':
                try:
                    setframe = int(values['sliderframe'])
                    cap.set(cv2.CAP_PROP_POS_FRAMES, setframe)
                    ret, frame = cap.read()
                    if ret:
                        window['nowframe'].update(setframe)
                        movieinfo = '/' + str(allframe) + ';(' + str(setframe / fps) + ' s)'
                        window['movieinfo'].update(movieinfo)
                except:
                    print('err')
                    continue
            
            if event == 'nowframe':
                nowframe = int(values['nowframe'])
                cap.set(cv2.CAP_PROP_POS_FRAMES, nowframe)
                ret, frame = cap.read()

            if 'setroi' in event and not values['selectbeads']:
                if layoutlist[0][1].status == False and layoutlist[1][1].status == False:
                    if 'gel' in event:
                        NAME,DATA = layoutlist[0]
                    if 'bead' in event:
                        NAME,DATA = layoutlist[1]
                    window['roitext'+NAME].update('ON')
                    DATA.status = True

                else:
                    NAME,DATA = layoutlist[0]
                    SUBNAME,SUBDATA = layoutlist[1]
                    if 'gel' in event:
                        pass
                    if 'bead' in event:
                        NAME, SUBNAME = SUBNAME, NAME
                        DATA, SUBDATA = SUBDATA, DATA

                    DATA.status = not DATA.status
                    if DATA.status:
                        window['roitext'+NAME].update('ON')
                        SUBDATA.status = not SUBDATA.status
                        window['roitext'+SUBNAME].update('OFF')
                    else:
                        window['roitext'+NAME].update('OFF')

            if 'thtext' in event:
                for l in layoutlist:
                    NAME, DATA = l
                    try:
                        val = int(values['thtext'+NAME])
                        if val < 0:
                            val = 0
                        elif val > 255:
                            val = 255

                    except:
                        window['thtext'+NAME].update('')
                        continue

                    window['thinput'+NAME].update(val)
                    window['thtext'+NAME].update(val)
                    DATA.th = val

            if 'thinput' in event:
                for l in layoutlist:
                    NAME, DATA = l
                    val = int(values['thinput'+NAME])
                    window['thtext'+NAME].update(val)
                    DATA.th = val

            if event == 'selectbeads':
                if values['selectbeads']:
                    for l in layoutlist:
                        NAME,DATA = l
                        DATA.status = False
                        window['roitext'+NAME].update('OFF')


            if event == 'start':
                try:
                    startframe = int(values['startframe'])
                    endframe = int(values['endframe'])
                    bar_max = endframe - startframe
                    cap.set(cv2.CAP_PROP_POS_FRAMES, startframe)
                    window['-PROG-'].update(0, max = bar_max)

                    if values['selectfolder'] == '':
                        filename = os.path.splitext(filepath)[0]
                    else:
                        filename = values['selectfolder'] + '/' + os.path.splitext(movietitle)[0]
                    
                    myfn.SaveDatalist(window, values, cap, ret_scale, filename, fps, width, height, bar_max, layoutlist, startframe, contrast, brightness)
                    cv2.destroyAllWindows()
                    sg.popup('Analysis End')

                except Exception as e:
                    print(e.args)
                    print('Analysis Error')

            for n in imgp_arr:
                if n['name'] in event:
                    try:
                        if 'input' in event:
                            val = float(values['input' + n['name']])
                            window['slider' + n['name']].update(val)
                        elif 'slider' in event:
                            val = float(values['slider' + n['name']])
                            window['input' + n['name']].update(val)

                        if 'Contrast' in event:
                            contrast = val
                        elif 'Brightness' in event:
                            brightness = val
                        elif 'Gamma' in event:
                            gamma = val
                    except:
                        print('Error')

            if event == 'imgp_default':
                try:
                    contrast = 1.0
                    brightness = 0
                    gamma = 1.0
                    for n in imgp_arr:
                        window['input' + n['name']].update(n['val'])
                        window['slider' + n['name']].update(n['val'])
                except:
                    print('error')

        except Exception as e:
            print(e.args)
            print('Error')

        finally:
            if not frame is None:
                img, ret_scale = resize(frame, (1280, 720))
                img = img * contrast + brightness
                img = np.clip(img, 0, 255).astype(np.uint8)
                # gamma 
                #table = (np.arange(256)/255) ** gamma * 255
                #table = np.clip(table, 0, 255).astype(np.uint8)
                #img = cv2.LUT(img, table)

                binimg = img.copy()

                for l in layoutlist:
                    NAME, DATA = l
                    if len(DATA.roi) > 0:
                        if len(DATA.roi) == 2:
                            if values['Threshold'+NAME]:
                                img_crop = myfn.ROIimg(binimg, DATA.roi.astype(np.int32))
                                img_crop = myfn.BinConv(img_crop, int(values['thinput'+NAME]))
                                if 'gel' in NAME:
                                    kernel = np.ones((3,3), np.uint8)
                                    img_crop = cv2.dilate(img_crop, kernel, iterations=1)
                                if values['Posi'+NAME]:
                                    img_crop = cv2.bitwise_not(img_crop)

                                cv2.imshow(NAME, img_crop)
                            else:
                                cv2.destroyWindow(NAME)

                        if NAME == 'gel':
                            color = (255,0,0)
                        else:
                            color = (0,255,0)
                        
                        if len(DATA.roi) == 1:
                            img = cv2.rectangle(img, tuple(DATA.roi[0].astype(np.int32)), tuple(DATA.temppt.astype(np.int32)), color, thickness=2)
                        else:
                            img = cv2.rectangle(img, tuple(DATA.roi[0].astype(np.int32)), tuple(DATA.roi[1].astype(np.int32)), color, thickness=2)

                    if len(DATA.basepoints) > 0:
                        for p in DATA.basepoints:
                            img = cv2.circle(img, tuple(p.astype(np.int32)), 1, (0,0,255), 2)

                showimage(img, window)

    window.close()
    return result

if __name__ == '__main__':
    MainMenu()