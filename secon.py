import joblib
from skimage.io import imread
import numpy as np
from matplotlib import pyplot as plt

logreg=joblib.load("D:/Codes/computer_vision/logistic/archive/data/vehicle_mode.pkl")

img_path="D:/Codes/computer_vision/logistic/archive/data/vehicles/10.png"
img_two_path="D:/Codes/computer_vision/logistic/archive/data/non-vehicles/extra178.png"

img=imread(img_path)
img_two=imread(img_two_path)

img_flattened=img.reshape(1,-1)

img_two_flattened=img_two.reshape(1,-1)

pred=logreg.predict(img_flattened)

pred_two=logreg.predict(img_two_flattened)

fig,axs=plt.subplots(1,2,figsize=(10,5))

axs[0].imshow(img)
axs[0].set_title(("Vehicle" if pred[0]==1 else "Non -vehicle"))
axs[0].axis("off")

axs[1].imshow(img_two)
axs[1].set_title(("Vehicle" if pred_two[0]==1 else "Non -vehicle"))
axs[1].axis("off")

plt.show()
