from getImages import *
import matplotlib.pyplot as plt
import numpy as np

home = "../input/"
puzzle1 = 'puzzle1/'
puzzle2 = 'puzzle2/'



def rescale(array):
    array = (array - np.min(array)) / (np.max(array) - np.min(array))
    array = array / np.max(array)
    return array


def templateMatching(list):
    map = list[0].copy()
    template = list[1].copy()
    m_h, m_w = map.shape[0], map.shape[1]
    t_h, t_w = template.shape[0], template.shape[1]
    ssd_h, ssd_w = m_h - t_h, m_w - t_w
    map = rescale(map.astype(np.single))
    template = rescale(template.astype(np.single))
    ssd = np.zeros((ssd_h, ssd_w))
    for i in range(ssd_h):
        for j in range(ssd_w):
            ssd[i, j] = np.sum((map[i:i + t_h, j:j + t_w] - template) ** 2)
    ssd = rescale(ssd.astype(np.single))
    (_, _, minLoc, _) = cv2.minMaxLoc(ssd)
    topLeft = minLoc
    botRight = (topLeft[0] + t_w, topLeft[1] + t_h)
    roi = map[topLeft[1]:botRight[1], topLeft[0]:botRight[0]]
    mask = np.zeros(map.shape, dtype="single")
    map = cv2.addWeighted(map, 0.25, mask, 0.75, 0)
    map[topLeft[1]:botRight[1], topLeft[0]:botRight[0]] = roi
    return (map * 255).astype(np.uint8)


def plot_image(im, title, cv=True, xticks=[], yticks=[]):
    plt.figure()
    if cv:
        plt.imshow(im[:, :, ::-1])
    else:
        plt.imshow(im)

    plt.title(title)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.show()


p1 = getImages(home + puzzle1)
p2 = getImages(home + puzzle2)

list_of_puzzles = [p1, p2]
save = "../output/"

for i in range(len(list_of_puzzles)):
  map_result = templateMatching(list_of_puzzles[i])
  cv2.imshow("Here's Waldo", map_result)
  cv2.imwrite(save + "Puzzle_" + str(i+1) + ".jpg", map_result)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

# map_result = templateMatching(p2)
# cv2.imshow("Map", map_result)
# cv2.waitKey(0)
