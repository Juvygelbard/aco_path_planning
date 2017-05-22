import matplotlib.pyplot as plt
import numpy as np
from itertools import product

# map config
SQR_MAP = 20

# algorithm config
ALPHA = 1
BETA = 2.5
EVAPORATION = 0.2
ANTS = 5

# simulation config
SHOW_PHEROMONES = 0
ITERATIONS_CLEAN = 100
ITERATIONS_OBSTICLE = 100


class SimpleMap:
    dir = ((-1, 0),
           (0, 1),
           (1, 0),

           (0, -1))

    def __init__(self, map, heuristic=lambda y_s, x_s, y_t, x_t: 0):
        self.map = map
        self.hw = np.shape(map)
        self.h, self.w = self.hw
        self.p_map = np.ones(self.hw) * 0.1
        self.heuristic = heuristic
        self.start = (0, 0)
        self.end = (self.h-1, self.w-1)

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

    def prob_numinator(self, dst_y, dst_x):
        return np.power(self.p_map[dst_y, dst_x], ALPHA) +\
               np.power(self.heuristic(dst_y, dst_x), BETA) # pheromone-component^alpha + heuristic-component^beta

    def run(self, ants, iterations, show_p=0):
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

                # walk until you've reached the end
                y, x = self.start
                while not (y == self.end[0] and x == self.end[1]):
                    # get probabilities for next step
                    prob = self.adj(y, x)
                    if y > 0:
                        prob[0] *= self.prob_numinator(y-1, x) # up
                    if x < self.w-1:
                        prob[1] *= self.prob_numinator(y, x+1) # right
                    if y < self.h-1:
                        prob[2] *= self.prob_numinator(y+1, x) # down
                    if x > 0:
                        prob[3] *= self.prob_numinator(y, x-1) # left

                    # normalize
                    prob /= np.sum(prob)

                    # save last step (fur heuristic 2)
                    last_y, last_x = y, x

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

            if show_p>0 and i%show_p==0:
                plt.imshow(self.p_map, interpolation="nearest")
                plt.show()

            # evaporate pheromones
            self.p_map *= (1 - EVAPORATION)

            # update new pheromones
            for m in range(ants):
                for y, x in paths[m]:
                    self.p_map[y, x] += 1/lengths[m]

            print(i, best_len)
        return best_path

    def add_obstacles_global(self, obs):
        self.p_map = np.ones(self.hw) * 0.1
        for o in obs:
            for y, x in product(range(o.ul_y, o.lr_y), range(o.ul_x, o.lr_x)):
                self.map[y, x] = 1
                self.p_map[y, x] = 0

    def add_obstacles_local(self, obs, max_dist=4):
        max_p = np.max(self.p_map)
        for o in obs:
            for y, x in product(range(self.h), range(self.w)):
                d = o.get_dist(y, x)
                if d == 0: # in obsticle
                    self.map[y, x] = 1
                    self.p_map[y, x] = 0
                elif d <= max_dist:
                    self.p_map[y, x] = max_p / 2 ** d


class Obstacle:
    def __init__(self, ul_y, ul_x, lr_y, lr_x):
        self.ul_y = ul_y
        self.ul_x = ul_x
        self.lr_y = lr_y
        self.lr_x = lr_x

    def get_dist(self, y, x):
        # case given point inside object
        if self.ul_y <= y <= self.lr_y and self.ul_x <= x <= self.lr_x:
            return 0

        # find reference y
        if self.ul_y <= y <= self.lr_y:
            ref_y = y
        else:
            ref_y = self.ul_y if abs(y-self.ul_y) < abs(y-self.lr_y) else self.lr_y

        # find reference x
        if self.ul_x <= x <= self.lr_x:
            ref_x = x
        else:
            ref_x = self.ul_x if abs(x-self.ul_x) < abs(x-self.lr_x) else self.lr_x

        # return distance to reference point = l1 norm
        return abs(y-ref_y) + abs(x-ref_x)


def display_path(map, path):
    d_map = np.array(map.map)
    for y, x in path:
        d_map[y, x] = 0.5

    plt.imshow(d_map, interpolation="nearest", cmap="gray_r")
    plt.show()

if __name__ == '__main__':
    # init maps
    raw_map = np.zeros((SQR_MAP, SQR_MAP))

    # init heuristics
    h1 = lambda y_d, x_d: y_d+x_d
    last_x, last_y = (0, 0)
    h2 = lambda y_d, x_d: 1 if y_d == last_y and x_d == last_x else 0.5

    # start simulation with empty map
    smap = SimpleMap(raw_map, heuristic=h1)
    path = smap.run(ANTS, ITERATIONS_CLEAN, show_p=SHOW_PHEROMONES)
    display_path(smap, path)

    # add obsticle
    o1 = Obstacle(10, 0, 11, 15)
    smap.add_obstacles_global([o1])
    path = smap.run(ANTS, ITERATIONS_OBSTICLE, show_p=SHOW_PHEROMONES)
    display_path(smap, path)
