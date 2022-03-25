import sys

import math, random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pyclipper


def return_Map(index=1):
    '''10m x 10m area'''
    boundary_coords = [(0, 0), (10, 0), (10, 10), (0, 10)] # anti-clockwise

    if index == 1:
        obstacle_list = [ 
            [(0.0, 0.0), (0.0, 10.0), (2.0, 10.0), (2.0, 0.0)],
            [(8.0, 0.0), (8.0, 10.0), (10.0, 10.0), (10.0, 0.0)],
            [(4.0, 4.0), (4.0, 6.0), (6.0, 6.0), (6.0, 4.0)], # middle obstacle
        ]
        nchoices = 2

    elif index == 2:
        obstacle_list = [ 
            [(0.0, 0.0), (0.0, 10.0), (2.0, 10.0), (2.0, 0.0)],
            [(8.0, 0.0), (8.0, 10.0), (10.0, 10.0), (10.0, 0.0)],
            [(2.0, 0.0), (2.0, 4.0), (6.0, 4.0), (6.0, 0.0)],
            [(4.0, 6.0), (4.0, 10.0), (8.0, 10.0), (8.0, 6.0)],
        ]
        nchoices = 1

    elif index == 3:
        obstacle_list = [ 
            [(0.0, 0.0), (0.0, 4.0), (4.0, 4.0), (4.0, 0.0)],
            [(6.0, 0.0), (6.0, 4.0), (10.0, 4.0), (10.0, 0.0)],
            [(2.0, 6.0), (2.0, 10.0), (4.0, 10.0), (4.0, 6.0)],
            [(6.0, 6.0), (6.0, 10.0), (8.0, 10.0), (8.0, 6.0)],
            [(0.0, 4.0), (0.0, 10.0), (0.1, 10.0), (0.1, 4.0)],
            [(9.9, 4.0), (9.9, 10.0), (10.0, 10.0), (10.0, 4.0)],
        ]
        nchoices = 3

    elif index == 4:
        obstacle_list = [ 
            [(0.0, 0.0), (0.0, 4.0), (4.0, 4.0), (4.0, 0.0)],
            [(0.0, 6.0), (0.0, 10.0), (4.0, 10.0), (4.0, 6.0)],
            [(6.0, 0.0), (6.0, 10.0), (10.0, 10.0), (10.0, 0.0)],
        ]
        nchoices = 3

    elif index == 5:
        obstacle_list = [ 
            [(0.0, 2.0), (0.0, 10.0), (4.0, 10.0), (4.0, 4.0)],
            [(2.0, 0.0), (6.0, 4.0), (10.0, 4.0), (10.0, 0.0)],
            [(6.0, 6.0), (6.0, 10.0), (10.0, 10.0), (10.0, 6.0)],
        ]
        nchoices = 3

    elif index == 6:
        obstacle_list = [ 
            [(0.0, 0.0), (0.0, 4.0), (10.0, 4.0), (10.0, 0.0)],
            [(2.0, 4.0), (5.0, 6.0), (8.0, 4.0)],
            [(0.0, 6.0), (0.0, 10.0), (8.0, 10.0), (2.0, 6.0)],
            [(6.5, 7.0), (10.0, 9.5), (10.0, 6.0), (8.0, 6.0)],
        ]
        nchoices = 3

    elif index == 7:
        obstacle_list = [ 
            [(0.0, 0.0), (0.0, 6.0), (6.0, 0.0)],
            [(4.0, 10.0), (10.0, 10.0), (10.0, 4.0)],
            [(2.0, 7.0), (3.0, 8.0), (5.5, 7.0), (3.0, 4.5)], # obstacle
        ]
        nchoices = 2

    elif index == 8:
        obstacle_list = [ 
            [(0.0, 0.0), (0.0, 10.0), (2.0, 10.0), (2.0, 0.0)],
            [(4.0, 0.0), (4.0, 10.0), (6.0, 10.0), (6.0, 0.0)],
            [(9.9, 0.0), (9.9, 10.0), (10.0, 10.0), (10.0, 0.0)],
            [(7.0, 4.0), (7.0, 6.0), (9.0, 6.0), (9.0, 4.0)], # obstacle
        ]
        nchoices = 3

    elif index == 9:
        obstacle_list = [ 
            [(0.0, 0.0), (0.0, 10.0), (2.0, 10.0), (2.0, 0.0)],
            [(2.0, 0.0), (2.0, 0.1), (10.0, 0.1), (10.0, 0.0)],
            [(2.0, 9.9), (2.0, 10.0), (10.0, 10.0), (10.0, 9.9)],
            [(4.0, 2.0), (4.0, 6.0), (8.0, 2.0)],
            [(4.0, 8.0), (10.0, 8.0), (10.0, 2.0)],
        ]
        nchoices = 2

    ### More for test
    elif index == 10:
        obstacle_list = [ 
            [(0.0, 0.0), (0.0, 10.0), (0.1, 10.0), (0.1, 0.0)],
            [(6.0, 0.0), (6.0, 10.0), (10.0, 10.0), (10.0, 0.0)],
            [(2.0, 4.0), (2.0, 6.0), (4.0, 6.0), (4.0, 4.0)], # obstacle
        ]
        nchoices = 2

    elif index == 11:
        obstacle_list = [ 
            [(0.0, 0.0), (0.0, 10.0), (4.0, 10.0), (4.0, 0.0)],
            [(6.0, 0.0), (6.0, 4.0), (10.0, 4.0), (10.0, 0.0)],
            [(6.0, 6.0), (6.0, 10.0), (8.0, 10.0), (8.0, 6.0)],
            [(9.9, 4.0), (9.9, 10.0), (10.0, 10.0), (10.0, 4.0)],
        ]
        nchoices = 2

    elif index == 12:
        obstacle_list = [ 
            [(8.0, 10.0), (10.0, 10.0), (10.0, 2.0), (4.5, 4.5)],
            [(10.0, 0.0), (0, 0), (3, 3)],
            [(0.0, 2.5), (0, 10), (6, 10)],
        ]
        nchoices = 2

    else:
        raise IndexError("Index doesn't exist.")

    return boundary_coords, obstacle_list, nchoices

def get_ref_path(index=1, choice=1, reverse=False):
    path = None

    if index == 1:
        if choice == 1:
            path = [(5, 0), (5, 3), (3, 4), (3, 6), (5, 7), (5, 10)]
        elif choice > 1:
            path = [(5, 0), (5, 3), (7, 4), (7, 6), (5, 7), (5, 10)]
    elif index == 2:
        path = [(7, 0), (7, 5), (3, 5), (3, 10)]
    elif index == 3:
        if choice == 1:
            path = [(5, 0), (5, 10)]
        elif choice == 2:
            path = [(5, 0), (5, 5), (1, 5), (1, 10)]
        elif choice > 2:
            path = [(5, 0), (5, 5), (9, 5), (9, 10)]
    elif index == 4:
        if choice == 1:
            path = [(5, 0), (5, 10)]
        elif choice == 2:
            path = [(0, 5), (5, 5), (5, 10)]
        elif choice > 2:
            path = [(0, 5), (5, 5), (5, 0)]
    elif index == 5:
        if choice == 1:
            path = [(0, 0), (5, 4), (5, 10)]
        elif choice == 2:
            path = [(0, 0), (6, 5), (10, 5)]
        elif choice > 2:
            path = [(10, 5), (5, 5), (5, 10)]
    elif index == 6:
        if choice == 1:
            path = [(0, 5), (2, 5), (9, 10)]
        elif choice == 2:
            path = [(0, 5), (2, 5), (5, 7), (8, 5), (10, 5)]
        elif choice > 2:
            path = [(9, 10), (5, 7), (8, 5), (10, 5)]
    elif index == 7:
        if choice == 1:
            path = [(0, 10), (2.5, 4), (5, 5), (10, 0)]
        elif choice > 1:
            path = [(0, 10), (6, 7.5), (5, 5), (10, 0)]
    elif index == 8:
        if choice == 1:
            path = [(3, 0), (3, 10)]
        elif choice == 2:
            path = [(8, 0), (8, 3), (6.5, 4), (6.5, 6), (8, 7), (8, 10)]
        elif choice > 2:
            path = [(8, 0), (8, 3), (9.5, 4), (9.5, 6), (8, 7), (8, 10)]
    elif index == 9:
        if choice == 1:
            path = [(10, 1), (3, 1), (3, 9), (10, 9)]
        elif choice > 1:
            path = [(10, 1), (3, 8), (3, 9), (10, 9)]

    ### More for test
    elif index == 10:
        if choice == 1:
            path = [(3, 0), (3, 3), (1, 4), (1, 6), (3, 7), (3, 10)]
        elif choice > 1:
            path = [(3, 0), (3, 3), (5, 4), (5, 6), (3, 7), (3, 10)]
    elif index == 11:
        if choice == 1:
            path = [(5, 0), (5, 10)]
        elif choice > 1:
            path = [(5, 0), (5, 5), (9, 5), (9, 10)]
    elif index == 12:
        if choice == 1:
            path = [(10, 1), (3, 4.2), (0, 1.5)]
        elif choice == 2:
            path = [(10, 1), (3, 4.2), (7, 10.0)]
        elif choice > 2:
            path = [(0, 1.5), (7, 10.0)]

    if path is None:
        raise KeyError('Invalid arguments.')
    if reverse:
        path = path[::-1]
    return path

def plot_path(path, ax, color='k--'):
    for i in range(len(path)-1):
        ax.plot([path[i][0], path[i+1][0]], [path[i][1], path[i+1][1]], color)

class MovingObject():
    def __init__(self, current_position, stagger=0):
        self.stagger = stagger
        self.traj = [current_position]

    def motion_model(self, ts, state, action):
        x,y = state[0], state[1]
        vx, vy = action[0], action[1]
        x += ts*vx
        y += ts*vy
        return (x,y)

    def one_step(self, ts, action):
        self.traj.append(self.motion_model(ts, self.traj[-1], action))

    def run(self, path, ts=.2, vmax=0.5):
        coming_path = path[1:]
        while(len(coming_path)>0):
            stagger = random.randint(0,10)/10*self.stagger
            x, y = self.traj[-1][0], self.traj[-1][1]
            dist_to_next_goal = math.hypot(coming_path[0][0]-x, coming_path[0][1]-y)
            if dist_to_next_goal < (vmax*ts):
                coming_path.pop(0)
                continue
            else:
                if random.randint(0,1)>0.5:
                    stagger = -stagger
                dire = ((coming_path[0][0]-x)/dist_to_next_goal, (coming_path[0][1]-y)/dist_to_next_goal)
                action = (dire[0]*math.sqrt(vmax)+stagger, dire[1]*math.sqrt(vmax)+stagger)
                self.one_step(ts, action)

class Graph:
    def __init__(self, boundary_coords, obstacle_list, inflation=0):
        self.boundary_coords = boundary_coords # in counter-clockwise ordering
        self.obstacle_list = obstacle_list.copy() # in clock-wise ordering
        self.preprocess_obstacle_list = obstacle_list.copy()
        self.inflator = pyclipper.PyclipperOffset()
        for i in range(len(self.preprocess_obstacle_list)):
            self.preprocess_obstacle_list[i] = self.preprocess_obstacle(pyclipper.scale_to_clipper(obstacle_list[i]), pyclipper.scale_to_clipper(inflation))

    def preprocess_obstacle(self, obstacle, inflation):
        self.inflator.Clear()
        self.inflator.AddPath(obstacle, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
        inflated_obstacle = pyclipper.scale_from_clipper(self.inflator.Execute(inflation))[0]
        inflated_obstacle.reverse()
        return inflated_obstacle

    def plot_map(self, ax, clean=False, empty=False):
        boundary = self.boundary_coords + [self.boundary_coords[0]]
        boundary = np.array(boundary)
        if empty:
            ax.plot(boundary[:,0], boundary[:,1], 'white')
            ax.axis('equal')
            return
        for obs in self.obstacle_list:
            obs = np.array(obs)
            poly = patches.Polygon(obs, color='gray', zorder=0)
            ax.add_patch(poly)
        if not clean:
            ax.plot(boundary[:,0], boundary[:,1], 'k')
            for obs in self.preprocess_obstacle_list:
                obs_edge = obs + [obs[0]]
                xs, ys = zip(*obs_edge)
                ax.plot(xs,ys,'b')
        ax.set_xlim([min(boundary[:,0]), max(boundary[:,0])])
        ax.set_ylim([min(boundary[:,1]), max(boundary[:,1])])


if __name__ == '__main__':
    ts = 0.2

    index = 5
    choice = 2

    inflation = 0
    stagger = 0# 0.4   + (random.randint(0, 20)/10-1) * 0.2
    vmax = 1        + (random.randint(0, 20)/10-1) * 0.3

    boundary_coords, obstacle_list, nchoices = return_Map(index=index)
    graph = Graph(boundary_coords, obstacle_list, inflation=inflation)

    ref_path_list = [get_ref_path(index=index, choice=x, reverse=False) for x in range(1,nchoices+1)]
    start = ref_path_list[choice][0]

    obj = MovingObject(start, stagger)
    obj.run(ref_path_list[choice], ts, vmax)
    traj = obj.traj

    fig, ax = plt.subplots()
    # ------------------------
    ax.axis('off')
    ax.margins(0)
    graph.plot_map(ax, clean=True, empty=False)
    for ref_path in ref_path_list:
        plot_path(ref_path, ax, color='rx--')
    ax.plot(np.array(traj)[:,0],np.array(traj)[:,1],'k.')
    ax.set_aspect('equal', 'box')
    plt.tight_layout()
    # ------------------------
    plt.show()