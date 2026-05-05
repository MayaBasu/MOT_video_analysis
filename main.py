import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import scipy as s
from astropy.modeling import models, fitting

plt.rcParams['figure.figsize'] = [4, 3]
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.family'] = 'cmr10'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["axes.formatter.use_mathtext"] = True



def compute_temp(v_rms):
    #v_rms is given in units of pixels per frame,
    #we take 100 pixels = 1cm
    #one frame is roughly a millisecond
    #so in meters per second we have

    # pixels per frame * (1m)/(100*100 pixels) * (1000 frames)/s
    v_rms_m_per_s = ((v_rms/100)/100)*1000

    print(f"v_rms is {v_rms} in pixels per frame, or {v_rms_m_per_s} in meters per second")



video_directory_path = "/Users/mayabasu/Desktop/motvideos"
files = os.listdir(video_directory_path)

for filename in files:
    print(filename)







example = "10_1.avi"
example2 = "pulse_10ms_xp_1ms_0.avi"

video_8_1 = "8.avi"
video_8_2 = "8_2.avi"


video_12_1 = "12_1.avi"
video_12_2 = "12_2.avi"

video_100_1= "100.avi"
video_100_2= "100_2.avi"

video_100_repeat="pulse_redux_100ms_1ms.avi" #great!!


video_10_repeat = "pulse_10ms_xp_1ms_0 copy.avi"
video_10_repeat2 = "pulse_10ms_xp_1ms_0.avi"

video_5_repeat = "pulse_5ms_xp_1ms_2.avi" #has one!!

video_5_repeat2 = "pulse_5ms_xp_1ms.avi" #not good

video_5_repeat3 = "pulse_5ms_xp_1ms_0.avi" #not good

video_8_repeat_1 = "pulse_redux_8ms_1ms.avi" #(1000,3000)


video_8_repeat_2 = "pulse_8ms_xp_1ms_0.avi" # has big dips


def gaussian_fit(frame, plot):

    gaussian_model = models.Gaussian2D(x_mean=frame.shape[1]/2,y_mean= frame.shape[0]/2)
    fitter = fitting.LMLSQFitter()
    y, x = np.mgrid[:frame.shape[0], :frame.shape[1]]
    z = frame[y, x]
    fitted_gaussian = fitter(gaussian_model, x,y,z)
    #print(fitted_gaussian.y_fwhm, fitted_gaussian.x_fwhm)
    if plot:
        fig, axs = plt.subplots(figsize=(8, 2.5), ncols=3)
        axs[0].imshow(z)
        axs[0].set_title("Frame")
        axs[1].imshow(fitted_gaussian(x,y))
        axs[1].set_title("Model")
        axs[2].imshow(z - fitted_gaussian(x, y))
        axs[2].set_title("Residual")

    return np.sqrt(fitted_gaussian.y_fwhm**2+fitted_gaussian.x_fwhm**2)



fit_gaussian = False
def load(filename,trim, plot=True):
    path = os.path.join(video_directory_path, filename)
    video = cv.VideoCapture(path)
    num_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    print(f"Loading {filename} from {path}. Video length: {num_frames}")

    frame_time_stamps = []
    total_brightness = []
    widths = []

    for i in range(num_frames):
        succeeded, frame = video.read()
        frame_time_stamps.append(video.get(cv.CAP_PROP_POS_MSEC))
        if not succeeded:
            print(f"Failed to load frame number {i}")
            raise SystemExit(0)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if fit_gaussian:
                if i % 1 ==0:
                    print(f"fitting gaussian {i}")
                    widths.append(gaussian_fit(frame,False))
            total_brightness.append(np.sum(frame))


    if plot:
        fig, ax = plt.subplots(4)
        ax[0].plot(np.linspace(0, num_frames, num_frames), frame_time_stamps)
        ax[0].set_xlabel("Frame number")
        ax[0].set_ylabel("Video time (ms)")
        ax[0].set_title("Video time vs. Frame number")

        ax[1].plot(frame_time_stamps, total_brightness/max(total_brightness))
        ax[1].set_xlabel("Video time (ms)")
        ax[1].set_ylabel("Normalized brightness")
        ax[1].set_title("Brightness")

        ax[2].plot(frame_time_stamps, s.ndimage.gaussian_filter1d(total_brightness / max(total_brightness), sigma=5))
        ax[2].set_xlabel("Video time (ms)")
        ax[2].set_ylabel("Normalized brightness")
        ax[2].set_title("Brightness")
        trimmed = total_brightness[trim[0]:trim[1]]

        ax[3].plot(np.linspace(0, len(trimmed), len(trimmed)), trimmed)
        ax[3].set_xlabel("Frame number")
        ax[3].set_ylabel("$\sqrt{x_fwhm^2+y_fwhm^2}$")
        ax[3].set_title("Width of a fitted Gaussian")

        plt.tight_layout()
        plt.savefig(f"Figure_{filename}.png")

        plt.show()
    return frame_time_stamps, total_brightness


class Video:
    def __init__(self, filename, trims):
        self.filename = filename
        self.trims = trims
    def load(self):
        load(self.filename, [0,-1])
    def plot_trims(self):
        frame_time_stamps, total_brightness = load(self.filename,self.trims,plot=False)
        fig,ax = plt.subplots(nrows=1, ncols=len(self.trims))
        i = 0
        for trim in self.trims:
            trimmed = total_brightness[trim[0]:trim[1]]
            trim_length = len(trimmed)
            if len(self.trims) > 1:
                ax[i].scatter(np.linspace(1,trim_length,trim_length), trimmed/np.max(trimmed))
            else:
                ax.scatter(np.linspace(1,trim_length,trim_length), trimmed/np.max(trimmed))
            i += 1
        plt.show()
    def show_images(self,i,h,ax_ex):
        trim = self.trims[i]
        num = int(trim[1]-trim[0])
        rows = np.ceil(np.sqrt(num))
        cols = np.ceil(num/rows)
        fig,ax = plt.subplots(nrows=int(rows), ncols=int(cols))

        path = os.path.join(video_directory_path, self.filename)
        video = cv.VideoCapture(path)
        num_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
        j = 0
        saved_frames = []
        saved_frame_nums = []
        for i in range(num_frames):
            succeeded, frame = video.read()
            frame_time_stamp = video.get(cv.CAP_PROP_POS_MSEC)
            if not succeeded:
                print(f"Failed to load frame number {i}")
                raise SystemExit(0)
            else:
                if i in range(trim[0],trim[1]):
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    ax.flatten()[j].imshow(frame)
                    saved_frames.append(frame)
                    saved_frame_nums.append(i)
                    #ax[j].set_title(f"{frame_time_stamp}")
                    ax.flatten()[j].axis('off')
                    ax.flatten()[j].set_title("{}: {:.1f}".format(i,frame_time_stamp))
                    j += 1

      #  plt.tight_layout()
      #  plt.show()

        diameters = []


        for frame in saved_frames:
            diameters.append(gaussian_fit(frame, False))
        print(saved_frame_nums)
        print(diameters)
       # diameters = np.array(diameters)/diameters[0]

        shift = True
        if shift:
            saved_frame_nums = np.array(saved_frame_nums)-saved_frame_nums[0] + 1

        ax_ex.plot(saved_frame_nums[:h[0]], diameters[:h[0]])
        ax_ex.plot(saved_frame_nums[h[1]:], diameters[h[1]:])

        x0 = saved_frame_nums[h[0]-1]
        x1 = saved_frame_nums[h[1]]
        y0 = diameters[h[0]-1]
        y1 = diameters[h[1]]
        print(x0,x1,y0,y1)
        v_rms= (y1-y0)/(x1-x0)
        x = np.linspace(x0,x1,100)
        y = (x-x0)*v_rms + y0
        ax_ex.plot(x,y,linestyle='dashed',label=f"$v_rms {v_rms}$")
        print("temp")
        compute_temp(v_rms)
        ax_ex.legend()

        #plt.show()









pulse_5ms_xp_1ms_0 = "pulse_5ms_xp_1ms_0.avi" #no decernable features
pulse_5ms_xp_1ms_1 = "pulse_5ms_xp_1ms_1.avi" #very short

pulse_5ms_xp_1ms_2 = "pulse_5ms_xp_1ms_2.avi" #one dip!
video_pulse_5ms_xp_1ms_2 = Video(pulse_5ms_xp_1ms_2,[(1757,1771)])
#video_pulse_5ms_xp_1ms_2.load()
#video_pulse_5ms_xp_1ms_2.plot_trims()
#fig,ax = plt.subplots(nrows=1, ncols=1)
#video_pulse_5ms_xp_1ms_2.show_images(0,[4,10],ax)

#plt.show()


pulse_5ms_xp_1ms_3 = "pulse_5ms_xp_1ms_3.avi" #no decernable data



pulse_redux_5ms_1ms = "pulse_redux_5ms_1ms.avi" #two dips! really nice loading curve
video_pulse_redux_5ms_1ms = Video(pulse_redux_5ms_1ms,[(1793,1809),(3129,3151)])
#video_pulse_redux_5ms_1ms.load()
#video_pulse_redux_5ms_1ms.plot_trims()
#video_pulse_redux_5ms_1ms.show_images(1,[4,12])
#video_pulse_redux_5ms_1ms.show_images(0,[2,8])

pulse_8ms_xp_1ms_0 = "pulse_8ms_xp_1ms_0.avi" #two dips at the end!
video_pulse_8ms_xp_1ms_0 = Video(pulse_8ms_xp_1ms_0,[(1950,1980),(2527,2544)])
#video_pulse_8ms_xp_1ms_0.load()
#video_pulse_8ms_xp_1ms_0.plot_trims()

pulse_redux_8ms_1ms = "pulse_redux_8ms_1ms.avi"# three or four dips begining
video_pulse_redux_8ms_1ms = Video(pulse_redux_8ms_1ms,[(1081,1164),(2117,2220),(2730,2800)])
#video_pulse_redux_8ms_1ms.load()
#video_pulse_redux_8ms_1ms.plot_trims()

pulse_10ms_xp_1ms_0 = "pulse_10ms_xp_1ms_0.avi" #no decernable data

pulse_redux_100ms_1ms = "pulse_redux_100ms_1ms.avi" #great  dips!




#video = Video(video_8_repeat_1,[(1081,1164),(2117,2220),(2730,2800)])
#video.load()
#video.plot_trims()
#load(pulse_10ms_xp_1ms_0,(0,-1)) #


def make_expansion_plots():

    fig,ax = plt.subplots(ncols=1,nrows=3)

    plt.suptitle("5ms pulse")
    pulse_redux_5ms_1ms = "pulse_redux_5ms_1ms.avi"  # two dips! really nice loading curve
    video_pulse_redux_5ms_1ms = Video(pulse_redux_5ms_1ms, [(1791, 1809), (3126, 3151)])

    pulse_5ms_xp_1ms_2 = "pulse_5ms_xp_1ms_2.avi"  # one dip!
    video_pulse_5ms_xp_1ms_2 = Video(pulse_5ms_xp_1ms_2, [(1757, 1782)])



    video_pulse_5ms_xp_1ms_2.show_images(0, [4, 9], ax[0])
    video_pulse_redux_5ms_1ms.show_images(1, [6, 15],ax[1])
    video_pulse_redux_5ms_1ms.show_images(0, [5, 10],ax[2])
    plt.legend()
    plt.show()






make_expansion_plots()
