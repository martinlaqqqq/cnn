from pynq.overlays.base import BaseOverlay
from pynq.lib.video import *
import cv2

def show(img):
    base = BaseOverlay("base.bit")
    # camera (input) configuration
    frame_in_w = 1280
    frame_in_h = 720

    # monitor configuration: 640*480 @ 60Hz
    Mode = VideoMode(frame_in_w,frame_in_h,24)
    hdmi_out = base.video.hdmi_out
    hdmi_out.configure(Mode,PIXEL_BGR)
    hdmi_out.start()
    print("HDMI Initialized")
    # initialize camera from OpenCV
    img = cv2.resize(img,(frame_in_w,frame_in_h))

    while True:
        outframe = hdmi_out.newframe()
        outframe[:,:] = img[:,:]
        hdmi_out.writeframe(outframe)
        if(base.buttons[2].read()==1):
            break

    hdmi_out.stop()
    del hdmi_out
