import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import colorsys

from palette_creator import paletteFromImage



# P = paletteFromImage("/Users/nissim/Documents/Programmes python/ColorPalette_Maker/images/15-.jpg", mode = "hsv", n_components=8, debug = True)
# plt.imshow(P)
# plt.show()

P = [[(0.29284604998201624, 0.361581274486119, 0.3589103224155234), (0.4665024180943736, 0.41387836290102054, 0.3411549228562378), (0.8053249892777419, 0.7613912322786601, 0.642898459367412), (0.7558142468807303, 0.8191928037701771, 0.798852625600463), (0.8261656124181377, 0.7061299349026766, 0.728367640693685)]]
P_hsv = [[colorsys.rgb_to_hsv(e[0],e[1],e[2]) for e in P[0]]]

def dist_sq(x,y,n):
    return sum([(x[i]-y[i])*(x[i]-y[i]) for i in range(n)])

def closest_match(x,P):
    d = 1000000
    ret = None
    for col in P:
        d2 = dist_sq(x,col,2)
        if d2<d:
            d = d2
            ret = col
    return ret

path = "/Users/nissim/Documents/Programmes python/ColorPalette_Maker/images/15-.jpg"
img = mpimg.imread(path)
n,m,_ = (img.shape)
n_img = np.zeros((n,m,3))
for i in range(n):
    for j in range(m):
        e = img[i][j]/255.0
        n_img[i][j] = closest_match(colorsys.rgb_to_hsv(e[0],e[1],e[2]),P_hsv[0])
        n_img[i][j] = colorsys.hsv_to_rgb(n_img[i][j][0],n_img[i][j][1],n_img[i][j][2])
plt.imshow(n_img)
plt.show()


