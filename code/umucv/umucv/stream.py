import numpy as np
import cv2   as cv
from threading import Thread
from umucv.util import Clock, putText
import datetime
import time
import argparse



def mkStream(sz=None, dev='0', loop=False):
    if dev == 'picam':
        from picamera.array import PiRGBArray
        from picamera import PiCamera
        import time
        camera = PiCamera()
        if sz is None:
            sz=(640,480)
        camera.resolution = sz
        camera.framerate = 32
        rawCapture = PiRGBArray(camera, size=sz)
        time.sleep(0.1)
        stream = camera.capture_continuous(rawCapture, format="bgr", use_video_port=True)
        for gen in stream:
            x = gen.array
            rawCapture.truncate(0)
            yield x


    elif dev[:5] == 'glob:':
        import glob
        files = sorted(glob.glob(dev[5:]))
        for f in files:
            yield cv.imread(f,cv.IMREAD_COLOR)


    elif dev[:4] == 'dir:':
        import glob
        import time
        files = sorted(glob.glob(dev[4:]))
        images = [ cv.imread(f,cv.IMREAD_COLOR) for f in files ]
        if not images:
            images = [ np.random.randint(256, size=(480,640,3), dtype= np.uint8) ]
        k = 0
        n = len(images)

        def fun(event, x, y, flags, param):
            nonlocal k
            #print(event)
            if event == cv.EVENT_LBUTTONDOWN:
                k = (k+1)%n
            if event == cv.EVENT_RBUTTONDOWN:
                k = (k-1)%n

        cv.namedWindow(dev,cv.WINDOW_NORMAL)
        h,w = images[0].shape[:2]
        cv.resizeWindow(dev,int(200*w/h),200)
        cv.setMouseCallback(dev, fun)
        while True:
            time.sleep(0.1)
            auxshow = images[k].copy()
            putText(auxshow, files[k])
            cv.imshow(dev, auxshow)
            yield images[k].copy()


    elif dev[:5] == 'shot:':
        for frame in mkShot(dev[5:],debug=True):
            yield frame


    elif dev[:5] == 'mjpg:':
        try:
            import urllib.request as url
        except:
            import urllib as url
        stream=url.urlopen(dev[5:])
        bytes=b''
        okread = False
        while True:
            bytes+=stream.read(1024)
            if(len(bytes)==0):
                if loop and okread:
                    stream=url.urlopen(dev[5:])
                    bytes=b''
                else:
                    break
            else:
                a = bytes.find(b'\xff\xd8')
                b = bytes.find(b'\xff\xd9')
                if a!=-1 and b!=-1:
                    jpg = bytes[a:b+2]
                    bytes= bytes[b+2:]
                    i = cv.imdecode(np.fromstring(jpg, dtype=np.uint8),cv.IMREAD_COLOR)
                    okread = True
                    yield i


    else:
        if dev in ['0','1','2','3','4']:
            dev = int(dev)
        cap = cv.VideoCapture(dev)
        assert(cap.isOpened())
        if sz is not None:
            w,h = sz
            cap.set(cv.CAP_PROP_FRAME_WIDTH,w)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT,h)
        w   = cap.get(cv.CAP_PROP_FRAME_WIDTH)
        h   = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv.CAP_PROP_FPS)
        print(f'{w:.0f}x{h:.0f} {fps}fps')
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                if loop:
                    cap = cv.VideoCapture(dev)
                else:
                    break


def withKey(stream, t=1):
    pausa = False
    key = 0
    exit = False
    for frame in stream:
        while True:
            key = cv.waitKey(t) & 0xFF
            if key == 27:
                exit = True
                break
            if key == 32:
                pausa = not pausa
                if pausa:
                    frozen = frame.copy()            
            if pausa:
                yield key, frozen.copy()
            else:
                yield key, frame
            if key == ord('s'):
                fname = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                cv.imwrite(fname+'.png',frame)
            if not pausa: break
        if exit: break




def isize(s):
    k = s.find('x')
    return (int(s[:k]),int(s[k+1:]))

def sourceArgs(p):
    p.add_argument('--dev', type=str, default='0', help='image source')
    p.add_argument('--size', help='desired image size', type=isize, default=None)
    p.add_argument('--resize', help='force image size', type=isize, default=None)
    p.add_argument('--pause', help='frame by frame', action='store_true')
    p.add_argument('--loop', help='repeat video forever', action='store_true')


def autoStream(transf = lambda x: x):
    parser = argparse.ArgumentParser()
    sourceArgs(parser)
    args, _ = parser.parse_known_args()
    
    def resize(x):
        if args.resize is None:
            return x
        wd,hd = args.resize
        h, w  = x.shape[:2]
        if h!=hd or w!=wd:
            if hd==0:
                hd = int(h/w*wd)
            if wd==0:
                wd = int(w/h*hd)
            return cv.resize(x, (wd,hd))
        else:
            return x
    
    stream = transf( map(resize, mkStream(args.size, args.dev, args.loop) ) )
    return withKey(stream, 0 if args.pause else 1)



def mkShot(ip, user=None, password=None, timeout=0.1, retries=3, debug=False):
    import requests
    while True:
        img = None
        for _ in range(retries):
            try:
                t0 = time.time()
                if user is None:
                    imgResp = requests.get(ip, timeout=timeout)
                else:
                    auth = requests.auth.HTTPDigestAuth(user, password)
                    imgResp = requests.get(ip, timeout=timeout, auth=auth)
                img = cv.imdecode(np.array(bytearray(imgResp.content),dtype=np.uint8),cv.IMREAD_COLOR)
                t1 = time.time()
                if debug: print(f'{(t1-t0)*1000:.0f}ms')
                break
            except:
                if debug: print('timeout')
        else:
            break
        yield img



class Camera:
    def __init__(self, size=None, dev=None, debug=False, transf = lambda x: x):
        self.clock = Clock()
        parser = argparse.ArgumentParser()
        sourceArgs(parser)
        args, _ = parser.parse_known_args()
        if size is None: size = args.size
        if dev  is None: dev  = args.dev
        self.stream = transf(mkStream(size,dev))
        self.frame = None
        self.time  = 0
        self.goon  = True
        self.drop  = False
        
        def fun():
            while self.goon:
                if self.drop:
                    next(self.stream)
                    if debug:
                        print('Frame dropped: {:.0f}'.format(self.clock.time()))
                else:    
                    self.frame = next(self.stream)
                    t = self.clock.time()
                    dt = t - self.time
                    self.time = t
                    if debug:
                        print('Frame ready: {:.0f} ({:.0f})'.format(self.time,dt))
        
        t = Thread(target=fun,args=())
        t.start()
        while self.frame is None: pass
    
    def stop(self):
        self.goon = False 
        
