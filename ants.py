from scipy.ndimage import imread
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
from itertools import product
import numpy as np

ALPHA = 1
BETA = 2.5
EVAPORATION = 0.5

class Map:
    dir = ((-1, 0),
           (0, 1),
           (1, 0),
           (0, -1))

    def __init__(self, map, pheromone=0.1, heuristic=lambda y_s, x_s, y_t, x_t: 0):
        self.map = map
        self.hw = np.shape(map)
        self.h, self.w = self.hw
        self.p_map = self.init_pheomones(self.h, self.w, pheromone)
        self.heuristic = heuristic
        self.start = (0, 0)
        self.end = (self.h-1, self.w-1)

    def init_pheomones(self, h, w, p):
        p_mat = lil_matrix((h*w, h*w))
        for y, x in product(range(h), range(w)):
            adj = self.adj(y, x)
            if(adj[0]): # up
                p_mat[y*w+x, (y-1)*w+x] = p
            if(adj[1]): # right
                p_mat[y*w+x, w+x+1] = p
            if(adj[2]): # down
                p_mat[y*w+x, (y+1)*w+x] = p
            if(adj[3]): # left
                p_mat[y*w+x, w+x-1] = p
        return p_mat

    def adj(self, y, x):
        ans = np.zeros(4)
        if y>0 and self.map[y-1][x]==0: # up
            ans[0] = 1
        if x<self.w-1 and self.map[y][x+1]==0: # right
            ans[1] = 1
        if y<self.h-1 and self.map[y+1][x]==0: # down
            ans[2] = 1
        if x>0 and self.map[y][x-1]==0: # left
            ans[3] = 1
        return ans

    def prob_numerator(self, src_y, src_x, dst_y, dst_x):
        return np.power(self.p_map[src_y*self.w+src_x, dst_y*self.w+dst_x], ALPHA) +\
               np.power(self.heuristic(src_y, src_x, dst_y, dst_x), BETA) # pheromone-component^alpha + heuristic-component^beta

    def run(self, ants, iterations):
        # init path and path length
        best_len = np.inf
        best_path = None

        for i in range(iterations):
            # set start position
            paths = tuple(list() for l in range(ants))
            lengths = np.empty(ants)

            for m in range(ants):
                # a single ant walk
                paths[m].clear
                paths[m].append(self.start)
                lengths[m] = 0

                # walk until youv'e reached the end
                y, x = self.start
                while not (y == self.end[0] and x == self.end[1]):
                    # get probabilities for next step
                    prob = self.adj(y, x)
                    if y > 0:
                        prob[0] *= self.prob_numerator(y, x, y - 1, x) # up
                    if x < self.w-1:
                        prob[1] *= self.prob_numerator(y, x, y, x + 1) # right
                    if y < self.h-1:
                        prob[2] *= self.prob_numerator(y, x, y + 1, x) # down
                    if x > 0:
                        prob[3] *= self.prob_numerator(y, x, y, x - 1) # left

                    # normlize
                    prob /= np.sum(prob)

                    # pick next step
                    c = np.random.choice(4, p=prob)
                    d_y, d_x = self.dir[c]
                    y += d_y
                    x += d_x

                    # save current path
                    paths[m].append((y, x))
                    lengths[m] += 1

                # update best path
                if lengths[m]<best_len:
                    best_len = lengths[m]
                    best_path = paths[m][:]

            # evaporate pheromones
            for y, x in product(range(self.h), range(self.w)):
                adj = self.adj(y, x)
                if(adj[0]): # up
                    self.p_map[y*self.w+x, (y-1)*self.w+x] *= (1-EVAPORATION)
                if(adj[1]): # right
                    self.p_map[y*self.w+x, y*self.w+x+1] *= (1-EVAPORATION)
                if(adj[2]): # down
                    self.p_map[y*self.w+x, (y+1)*self.w+x] *= (1-EVAPORATION)
                if(adj[3]): # left
                    self.p_map[y*self.w+x, y*self.w+x-1] *= (1-EVAPORATION)

            # update new pheromones
            for m in range(ants):
                for (y_src, x_src), (y_dst, x_dst) in zip(paths[m][:-1], paths[m][1:]):
                    self.p_map[y_src*self.w+x_src, y_dst*self.w+x_dst] += 1/lengths[m]

            print(i, best_len)
        return best_path


class obsticle:
    def __init__(self, ul_y, ul_x, lr_y, lr_x):
        self.ul_y = ul_y
        self.ul_x = ul_x
        self.lr_y = lr_y
        self.lr_x = lr_x


SQR_MAP = 20

# raw_map = imread("map1.png", flatten=True)
# for y, x in product(range(40), range(40)):
#     if raw_map[y, x] == 255:
#         raw_map[y, x] = 0
#     else:
#         raw_map[y, x] = 1

raw_map = np.zeros((SQR_MAP, SQR_MAP))
raw_map[5] = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1])
raw_map[15] = np.array([1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

print("building map...")
h = lambda y_s, x_s, y_d, x_d: y_d+x_d
map = Map(raw_map, heuristic=h)
print("done!")
path = map.run(5, 10)
for y, x in path:
    raw_map[y, x] = 0.5

plt.imshow(raw_map, interpolation="nearest")
plt.show()