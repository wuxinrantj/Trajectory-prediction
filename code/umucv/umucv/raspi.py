
from umucv.rpi.Adafruit_PWM_Servo_Driver import PWM

def PTU(chp = 0, cht = 1):
    global _p,_t
    pwm = PWM(0x40)
    pwm.setPWMFreq(50)
    _p = 0
    _t = 0
    smin = 180
    smax = 400
    smed = (smax+smin)/2
    sran = (smax-smin)/2
    def f(r):
      z = min(max(-1,r),1)
      return int(smed - z*sran)
    def ptu(x,y):
        global _p,_t
        _p = x
        _t = y
        #print(p,t)
        pwm.setPWM(chp, 0, f(_p))
        pwm.setPWM(cht, 0, f(_t))
    def dptu(dx,dy):
        global _p,_t
        ptu(_p+dx,_t+dy)
        _p = min(max(-1,_p),1)
        _t = min(max(-1,_t),1)
        return _p,_t
    def forward(l,r):
        pwm.setPWM(14, 0, r*4)
        pwm.setPWM(15, 0, l*4)


    ptu(0,0)
    
    return ptu,dptu,forward

