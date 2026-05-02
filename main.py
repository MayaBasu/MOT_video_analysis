import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import scipy as s
from astropy.modeling import models, fitting



video_directory_path = "/Users/mayabasu/Desktop/MOT_videos"
example = "10_1.avi"
example2 = "pulse_10ms_xp_1ms_0.avi"



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




def load(filename):
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
            if i % 1 ==0:
                print(f"fitting gaussian {i}")
                widths.append(gaussian_fit(frame,False))
            total_brightness.append(np.sum(frame))



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

    ax[3].plot(np.linspace(0,len(widths),len(widths)), widths)
    ax[3].set_xlabel("Frame number")
    ax[3].set_ylabel("$\sqrt{x_fwhm^2+y_fwhm^2}$")
    ax[3].set_title("Width of a fitted Gaussian")



    plt.tight_layout()
    plt.savefig(f"Figure_{filename}.png")



    plt.show()


load(example)



