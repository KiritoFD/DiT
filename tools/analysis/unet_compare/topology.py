"""Small topology measurements shared by the U-Net comparison and diagnostics."""
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, distance_transform_edt, label
from skimage.morphology import skeletonize


def analyze_image_pair(pred_path, gt_path):
    pred=np.asarray(Image.open(pred_path).convert('L'))/255 < .5
    gt=np.asarray(Image.open(gt_path).convert('L'))/255 < .5
    gl,gn=label(gt,np.ones((3,3)))
    pl,pn=label(pred,np.ones((3,3)))
    missing=sum(int((gl==i).sum())>=5 and float(pred[gl==i].mean())<.25 for i in range(1,gn+1))
    merged=0
    for i in range(1,pn+1):
        ids,counts=np.unique(gl[pl==i],return_counts=True)
        merged+=int(((ids>0)&(counts>=5)).sum()>1)
    thin=skeletonize(gt)&(distance_transform_edt(gt)<=2.5)
    return dict(thin_skeleton_recall=float(binary_dilation(pred)[thin].mean()) if thin.any() else 1.,
                missing_components=int(missing),merged_components=int(merged))
