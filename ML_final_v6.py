import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def sliding_window(image, stepSize):
    # slide a window across the image
    for x in range(0+2, image.shape[0]-2, stepSize):
        for y in range(0+2, image.shape[1]-2, stepSize):
            # yield the current window
            yield (x, y, image[x - 2:x + 3, y - 2:y + 3])


def denoise(Ct, m):
    winH = winW = m
    vCt = Ct.copy()
    t = np.zeros((2, Ct.shape[1]))
    t2 = np.zeros((Ct.shape[0] + 4, 2))
    Ct = np.vstack((t, Ct, t))
    Ct = np.column_stack((t2, Ct, t2))

    for (x, y, window) in sliding_window(Ct, stepSize=1):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        x -= 2
        y -= 2

        if np.count_nonzero(window == 1) / 25 >= nv:
            vCt[x, y] = 1
        else:
            vCt[x, y] = 0
    return vCt


def get_foreground(img_file, Mt, Vt, lam):
    img = np.array(cv2.imread(img_file), dtype=float)
    Dt = img - Mt
    result = (np.array(Dt ** 2 <= lam ** 2 * Vt).astype(int))
    Ct = np.zeros((result.shape[0], result.shape[1]))
    for i in range(0, result.shape[0]):
        for j in range(0, result.shape[1]):
            if not (result[i, j] == [1, 1, 1]).all():
                Ct[i, j] = 1
    vCt = denoise(Ct, 5)
    return vCt

# AirstripRunAroundFeb2006_1436(1).bmp,1437(1),1438(1),1439(1) these are duplicates delete it first from Images Folder

Image_files = glob.glob("/home/akshay/PycharmProjects/ML_assignment2/Images-20200603T173020Z-001/Images/*.bmp")
Image_files.sort()
Background_files = glob.glob('/home/akshay/PycharmProjects/ML_assignment2/BakSubGroundTruth-20200603T172937Z-001/BakSubGroundTruth/*.bmp')
Background_files.sort()

# initialization
lam = 1
lam_list = [0.5, 1, 2, 3, 4, 5, 6, 7]
nv = 0.6
tc = 0.01
vd = 9
m = 5

training_size=200 # take 500 for whole image sequence
test_sample_size=50
test_samples=np.random.randint(0,501,test_sample_size)
plt.figure("ROC Curve")
print("Running")
x_a=[]
y_a=[]

class_lables = []
class_lables = np.array(class_lables, dtype=int)
for f in test_samples:
    temp_arr = np.array(cv2.imread(Background_files[f])).ravel()[0::3]
    temp_arr[temp_arr > 0] = 1
    class_lables = np.concatenate(
        (class_lables, temp_arr),
        axis=None)

for lam in lam_list:
    im = cv2.imread(Image_files[0]) # First Image taken for initialization
    img = np.array(im, dtype=float)
    va = [vd, vd, vd]
    Vt = np.zeros(img.shape)
    Vt[:, :] = va
    Mt = img.copy()
    Mt = np.array(Mt, dtype=float)

    t = 1
    print("\nTraining Model For lambda = ",lam)
    for img_file in Image_files[1:training_size+1]: # 1st image is already processed
        print("Image Processed: ",t)
        img = np.array(cv2.imread(img_file), dtype=float)
        Dt = img - Mt
        result = (np.array(Dt ** 2 <= lam ** 2 * Vt).astype(int))
        Ct = np.zeros((result.shape[0], result.shape[1]))
        for i in range(0, result.shape[0]):
            for j in range(0, result.shape[1]):
                if (result[i, j] == [1, 1, 1]).all():
                    Ct[i, j] = 0
                else:
                    Ct[i, j] = 1

        vCt = denoise(Ct, m)
        alphat = lambda t: 1 / t if (1 / t) >= tc else tc
        t += 1
        del_vCt = np.zeros(Dt.shape)
        del_vCt_f = del_vCt.copy()
        for i in range(0, del_vCt.shape[0]):
            for j in range(0, del_vCt.shape[1]):
                if vCt[i, j] == 1:
                    del_vCt[i, j] = [0, 0, 0]
                    del_vCt_f[i, j] = [1, 1, 1]
                else:
                    del_vCt[i, j] = [1, 1, 1]
                    del_vCt_f[i, j] = [0, 0, 0]

        Mt = Mt + del_vCt * alphat(t) * Dt
        Vt = del_vCt_f * Vt + del_vCt * ((1 - alphat(t)) * (Vt + alphat(t) * Dt ** 2))


    pred_output = []
    pred_output = np.array(pred_output, dtype=int)

    print("\nProcessing Test Samples")

    for f in test_samples:
        print(".",end="")
        pred_output = np.concatenate(
            (pred_output, np.array(get_foreground(Image_files[f], Mt, Vt, lam), dtype=int) ), axis=None)
    fpr, tpr, _ = roc_curve(class_lables, pred_output, drop_intermediate=False)
    x_a.append(fpr[1])
    y_a.append(tpr[1])
    plt.plot(fpr[1], tpr[1],"s", label=str(lam))
    print("")

print("\nCheck ROC Curve")
plt.legend(title="lambda")
plt.plot(x_a,y_a,"k--",linewidth=1)
plt.xlabel("False Alarm Rate")
plt.ylabel("Sensitivity")
plt.show()
