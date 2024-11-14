import numpy as np
import cv2
from flask import Flask, send_file
from scipy.ndimage import gaussian_filter
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

def AIM(filename="/home/kiwi/documents/academy/kubernetes/knative-minio-sal-model/models/images/001.jpg", resizesize=1.0, convolve=False, thebasis='21jade950.mat', showoutput=True):
    # General setup and defaults
    sigval = 8
    sigwin = (30, 30)
    method = 1
    bins = 1000
    sigma = 0.01
    precision = 0.01
    psi = gaussian_filter(np.zeros((49, 49)), sigma=20)  # Create Gaussian filter for method 3

    scalingtype = 2
    dispthresh = 80
    contrastval = 6

    # Read and preprocess image
    print("Reading Image.")
    inimage = cv2.imread(filename, cv2.IMREAD_COLOR).astype(np.float32) / 255
    inimage = cv2.resize(inimage, (0, 0), fx=resizesize, fy=resizesize)

    # Load basis matrix
    print("Loading Basis.")
    basis_path = os.path.join(os.path.dirname(__file__), thebasis)
    basis_data = loadmat(basis_path)
    inbasis = basis_data['B']
    
    p = int(np.sqrt(inbasis.shape[1] / 3))
    pm = p - 1
    ph = pm // 2

    # Prepare projection and sparse representation
    print("Projecting local neighborhoods into basis space.")
    s = np.zeros((inbasis.shape[0], inimage.shape[0] - pm, inimage.shape[1] - pm))

    # Loop through patches in the image
    for i in range(ph, inimage.shape[0] - ph):
        for j in range(ph, inimage.shape[1] - ph):
            patch = inimage[i - ph:i + ph + 1, j - ph:j + ph + 1].reshape(-1)
            BVpatch = inbasis @ patch
            s[:, i - ph, j - ph] = BVpatch

    # Scale feature maps if needed
    minscale = s.min()
    maxscale = s.max()

    print("Performing Density Estimation.")
    ts = np.zeros_like(s)

    for z in range(s.shape[0]):
        tempim = s[z, :inimage.shape[0] - pm, :inimage.shape[1] - pm]
        if scalingtype == 2:
            minscale = tempim.min()
            maxscale = tempim.max()
        elif scalingtype == 3:
            # Load specific minmax values if available
            # minscale = learned_minval[z]
            # maxscale = learned_maxval[z]
            pass
        
        stempim = (tempim - minscale) / (maxscale - minscale)
        if method == 1:
            histo, _ = np.histogram(stempim, bins=bins, range=(0, 1))
            histo = histo / histo.sum()
            ts[z, :, :] = histo[(stempim * (bins - 1)).astype(int)] / histo.sum()
        elif method == 2:
            ts[z, :, :] = kernest(stempim, sigma, precision).reshape(tempim.shape)
        elif method == 3:
            # Method 3 logic for neighborhood based kernel density
            pass  # requires implementation of local density with kernel ψ

    print("Transforming likelihoods into information measures.")
    infomapt = -np.log(ts[0] + 1e-6)
    for z in range(1, s.shape[0]):
        infomapt -= np.log(ts[z] + 1e-6)

    if convolve:
        infomapt = gaussian_filter(infomapt, sigval)

    infomap = np.full(inimage.shape[:2], infomapt.min())
    infomap[ph:inimage.shape[0] - ph, ph:inimage.shape[1] - ph] = infomapt

    if showoutput:
        # Create the figure for the plots
        plt.figure(figsize=(10, 10))

        # Create the first subplot and save it
        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(inimage, cv2.COLOR_BGR2RGB))
        plt.title("Input Image")
        plt.savefig("output_image_1.png")  # Save this subplot as an image
        plt.clf()  # Clear the current figure to prevent overlap

        # Create the second subplot and save it
        plt.subplot(2, 2, 2)
        plt.imshow(infomap[sigwin[0]:-sigwin[0], sigwin[1]:-sigwin[1]], cmap='hot')
        plt.title("Information Map")
        plt.savefig("output_image_2.png")  # Save this subplot as an image
        plt.clf()

        # Create the third subplot and save it
        threshmap2 = np.minimum((infomap / np.percentile(infomap, 98)), 1)
        plt.subplot(2, 2, 3)
        plt.imshow(cv2.cvtColor(inimage, cv2.COLOR_BGR2RGB))
        plt.imshow(threshmap2 ** contrastval, cmap='hot', alpha=0.5)
        plt.title("Thresholded Image (Map 2)")
        plt.savefig("output_image_3.png")  # Save this subplot as an image
        plt.clf()

        # Create the fourth subplot and save it
        threshmap1 = infomap > np.percentile(infomap, dispthresh)
        tempim = np.zeros_like(inimage)
        for c in range(3):
            tempim[:, :, c] = threshmap1 * inimage[:, :, c]
        plt.subplot(2, 2, 4)
        plt.imshow(tempim)
        plt.title("Thresholded Image (Map 1)")
        plt.savefig("output_image_4.png")  # Save this subplot as an image
        plt.clf()


        
        out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images', '001_aim.png')
        plt.figure(figsize=(10, 10))
        plt.imshow(infomap[sigwin[0]:-sigwin[0], sigwin[1]:-sigwin[1]], cmap='gray')
        plt.savefig(out_path)
        print("Images have been saved as output_image_1.png, output_image_2.png, output_image_3.png, and output_image_4.png.")

    return infomap


def kernest(inmap, h, precision):
    # Determine the size of the input map (inmap)
    imsize = np.prod(inmap.shape)
    
    # Flatten the input array
    x = inmap.flatten()
    Nx = len(x)
    
    # Create the 'ax' array from 0 to 1 with the specified precision
    ax = np.arange(0, 1 + precision, precision)
    
    # Initialize the output 'y' array with zeros
    y = np.zeros_like(ax)
    
    # Loop through each element in x and update y
    for i in range(Nx):
        y += np.exp(-0.5 * ((ax - x[i]) ** 2) / (h ** 2))
    
    # Normalize y so that its elements sum to 1
    y = y / np.sum(y)
    
    # Return the distribution 'y'
    return y

@app.route('/', methods=['GET'])
def start_aim():
    inimg_path =  os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images', '001.jpg')
    AIM(filename=inimg_path)

    outimg_path =  os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images', '001_aim.png')
    return send_file(outimg_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)