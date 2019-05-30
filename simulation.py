import os
import argparse
import json
import time
import multiprocessing

import numpy as np

from classes import *


def random_obstacle(position1, position2, r):
    """Return an obstacle of radius r randomly placed between position1 and position2"""
    d = position1 - position2
    d_len = np.sqrt(d.dot(d))
    cos = d[0] / d_len
    sin = d[1] / d_len

    # Generat random x and y assuming d is aligned with x axis.
    x = np.random.uniform(2+r, d_len-r)
    y = np.random.uniform(-2*r, 2*r)

    # Rotate the alignment back to the actural d.
    true_x = x * cos + y * sin + position2[0]
    true_y = x * sin - y * cos + position2[1]

    return Sphere(r, [true_x, true_y], ndim=2)


def simulation(_):
    region = (-100, 100, -100, 100)

    env = Environment2D(region)

    for _ in range(ARGS.boids):
        agent = Boid(ndim=2, vision=ARGS.vision, size=ARGS.size,
                     max_speed=10, max_acceleration=20)
        agent.initialize(np.random.uniform(-80, 80, 2),
                         np.random.uniform(-15, 15, 2))
        env.add_agent(agent)
    for _ in range(ARGS.vicseks):
        agent = Vicsek(ndim=2, vision=ARGS.vision, size=ARGS.size,
                       max_speed=10, max_acceleration=20)
        agent.initialize(np.random.uniform(-80, 80, 2),
                         np.random.uniform(-15, 15, 2))
        env.add_agent(agent)

    goal = Goal(np.random.uniform(-40, 40, 2), ndim=2)
    env.add_goal(goal)
    # Create a sphere obstacle near segment between avg boids position and goal position.
    avg_boids_position = np.mean(
        np.vstack([agent.position for agent in env.population]), axis=0)

    spheres = []
    for _ in range(ARGS.obstacles):
        sphere = random_obstacle(avg_boids_position, goal.position, 8)
        spheres.append(sphere)
        env.add_obstacle(sphere)

    position_data = []
    velocity_data = []
    for _ in range(ARGS.steps):
        env.update(ARGS.dt)
        position_data.append([goal.position for goal in env.goals] +
                             [sphere.position for sphere in spheres] +
                             [agent.position.copy() for agent in env.population])
        velocity_data.append([np.zeros(2) for goal in env.goals] +
                             [np.zeros(2) for sphere in spheres] +
                             [agent.velocity.copy() for agent in env.population])

    return position_data, velocity_data


def main():
    if not os.path.exists(ARGS.save_dir):
        os.makedirs(ARGS.save_dir)

    with open(ARGS.config) as f:
        model_config = json.load(f)

    if ARGS.boids > 0:
        Boid.set_model(model_config["boid"])
    if ARGS.vicseks > 0:
        Vicsek.set_model(model_config["vicsek"])

    pool = multiprocessing.Pool(processes=ARGS.processes)
    position_data_all = []
    velocity_data_all = []

    instances = ARGS.instances
    batch = 100

    prev_time = time.time()
    while instances > 0:
        n = min(instances, batch)
        data_pool = pool.map(simulation, range(n))

        position_pool, velocity_pool = zip(*data_pool)

        instances -= n
        print('Simulation {}/{}... {:.1f}s'.format(ARGS.instances - instances,
                                                   ARGS.instances, time.time()-prev_time))
        prev_time = time.time()

        position_data_all.extend(position_pool)
        velocity_data_all.extend(velocity_pool)

    # print('Simulations {0}/{0} completed.'.format(ARGS.instances))

    np.save(os.path.join(ARGS.save_dir, ARGS.prefix +
                         '_position.npy'), position_data_all)
    np.save(os.path.join(ARGS.save_dir, ARGS.prefix +
                         '_velocity.npy'), velocity_data_all)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--boids', type=int, default=10,
                        help='number of boid agents')
    parser.add_argument('--vicseks', type=int, default=0,
                        help='number of vicsek agents')
    parser.add_argument('--obstacles', type=int, default=0,
                        help='number of obstacles')
    parser.add_argument('--vision', type=float, default=None,
                        help='vision range to determine range of interaction')
    parser.add_argument('--size', type=float, default=3,
                        help='agent size')
    parser.add_argument('--steps', type=int, default=200,
                        help='number of simulation steps')
    parser.add_argument('--instances', type=int, default=1,
                        help='number of simulation instances')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='time resolution')
    parser.add_argument('--config', type=str, default='config/default.json',
                        help='path to config file')
    parser.add_argument('--save-dir', type=str,
                        help='name of the save directory')
    parser.add_argument('--prefix', type=str, default='',
                        help='prefix for save files')
    parser.add_argument('--processes', type=int, default=1,
                        help='number of parallel processes')

    ARGS = parser.parse_args()

    main()
