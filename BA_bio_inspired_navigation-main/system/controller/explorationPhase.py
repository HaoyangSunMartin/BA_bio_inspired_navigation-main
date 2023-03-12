import numpy as np


def compute_exploration_goal_vector(env, i,trag_coding = "Full_Exploration"):
    """Determines goal vector where the agent should head to"""
    position = env.xy_coordinates[-1]
    env.goal_vector = env.goal - position

    distance_to_goal = np.linalg.norm(env.goal_vector)
    # Check if agent has reached the (sub) goal
    if distance_to_goal < 0.2 or i == 0:
        if env.env_model == "linear_sunburst":
            navigate_to_location(env, trag_coding = trag_coding)  # Pick next goal to travel tp
            # pick_random_location(env, rectangular=True)
        elif env.env_model == "single_line_traversal":
            if i == 0:
                pick_random_straight_line(env)  # Pick random location at the beginning
                # print("Heading to ", env.goal)
        else:
            pick_random_location(env)


def pick_random_straight_line(env):
    """Picks a location at circular edge of environment"""
    angle = np.random.uniform(0, 2 * np.pi)
    env.goal = env.xy_coordinates[0] + np.array([np.cos(angle), np.sin(angle)]) * env.arena_size


def pick_random_location(env, rectangular=False):
    if not rectangular:
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(0.5, 1) * env.arena_size
        env.goal = np.array([np.cos(angle), np.sin(angle)]) * distance
    else:
        x = np.random.uniform(0, 11)
        y = np.random.uniform(7, 11)
        env.goal = np.array([x, y])





def automated_exploration(env):
    print("calling automated Exploration module")

def get_exploration_trajectory(trag_coding = "Full_Exploration"):
    sll = [1.5, 1.5]
    slu = [1.5, 4.5]
    srl = [9.5, 1.5]
    sru = [9.5, 4.5]
    ing01 = [1.5, 4.5]
    outg01 = [1.5, 7.5]
    ing02 = [3.5, 4.5]
    outg02 = [3.5, 7.5]
    ing03 = [5.5, 4.5]
    outg03 = [5.5, 7.5]
    ing04 = [7.5, 4.5]
    outg04 = [7.5, 7.5]
    ing05 = [9.5, 4.5]
    outg05 = [9.5, 7.5]

    en01 = [0.5, 7.5]
    en02 = [2.5, 7.5]
    en03 = [4.5, 7.5]
    en04 = [6.5, 7.5]
    en05 = [8.5, 7.5]
    en06 = [10.5, 7.5]

    ex01 = [0.5, 10]
    ex02 = [2.5, 10]
    ex03 = [4.5, 10]
    ex04 = [6.5, 10]
    ex05 = [8.5, 10]
    ex06 = [10.5, 10]

    if trag_coding == "Full_Exploration":
        ##G01: [1.5, 4.5--7.5], G02: [3.5, 4.5--7.5], G03: [5.5, 4.5--7.5], G04: [7.5, 4.5--7.5], G05: [9.5, 4.5--7.5]
        goals = np.array([
            # sll,
            slu,
            # srl,
            sru,
            # sll,
            # srl,
            ing05,
            outg05,
            outg04,
            ing04,
            ing03,
            outg03,
            outg02,
            ing02,
            ing01,
            outg01,
            en01,
            ex01,
            ex02,
            en02,
            en03,
            ex03,
            ex04,
            en04,
            en05,
            ex05,
            ex06,
            en06,
            outg05,
            ing05

        ])
    elif trag_coding == "Target_First":
        goals = np.array([
            slu,
            ing01,
            outg01,
            en01,
            ex01,
            ex02,
            en02,
            outg02,
            ing02,
            ing03,
            outg03,
            en03,
            ex03,
            ex04,
            en04,
            outg04,
            ing04,
            ing05,
            outg05,
            en05,
            ex05,
            ex06,
            en06



        ])
    elif trag_coding=="simple_FULL":
        goals = np.array([
            [5.5, 4.5],
            [1.5, 4.5],
            [9.5, 4.5],
            [9.5, 7.5],
            [10.5, 7.5],
            [10.5, 10],
            [8.5, 10],
            [8.5, 7.5],
            [6.5, 7.5],
            [6.5, 10],
            [4.5, 10],
            [4.5, 7.5],
            [2.5, 7.5],
            [2.5, 10],
            [0.5, 10],
            [0.5, 7.5],
            [2.5, 7.5],
            [2.5, 10],
            [4.5, 10],
            [4.5, 7.5],
            [6.5, 7.5],
            [6.5, 10],
            [8.5, 10],
            [8.5, 7.5],
            [9.5, 7.5],

            outg04,
            ing04,
            ing03,
            outg03,
            outg02,
            ing02,
            ing01,
            outg01

        ])
    else:
        goals = np.array([
            [5.5, 4.5],
            [1.5, 4.5],
            [9.5, 4.5],
            [9.5, 7.5],
            [10.5, 7.5],
            [10.5, 10],
            [8.5, 10],
            [8.5, 7.5],
            [6.5, 7.5],
            [6.5, 10],
            [4.5, 10],
            [4.5, 7.5],
            [2.5, 7.5],
            [2.5, 10],
            [0.5, 10],
            [0.5, 7.5],
            [2.5, 7.5],
            [2.5, 10],
            [4.5, 10],
            [4.5, 7.5],
            [6.5, 7.5],
            [6.5, 10],
            [8.5, 10],
            [8.5, 7.5],
            [9.5, 7.5]
        ])
    """
        goals = np.array([
                 [5.5, 4.5],
                 [1.5, 4.5],
                 [9.5, 4.5],
                 [9.5, 7.5],
                 [10.5, 7.5],
                 [10.5, 10],
                 [8.5, 10],
                 [8.5, 7.5],
                 [6.5, 7.5],
                 [6.5, 10],
                 [4.5, 10],
                 [4.5, 7.5],
                 [2.5, 7.5],
                 [2.5, 10],
                 [0.5, 10],
                 [0.5, 7.5],
                 [2.5, 7.5],
                 [2.5, 10],
                 [4.5, 10],
                 [4.5, 7.5],
                 [6.5, 7.5],
                 [6.5, 10],
                 [8.5, 10],
                 [8.5, 7.5],
                 [9.5, 7.5]
                 ])
        """

    return goals

def navigate_to_location(env, trag_coding= "Full_Exploration"):
    """Pre-coded exploration path for linear sunburst maze"""
    goals = get_exploration_trajectory(trag_coding=trag_coding)

    idx = env.goal_idx
    if idx < goals.shape[0]:
        env.goal = goals[idx]
        print("Heading to goal: ", idx, env.goal)
        env.goal_idx = idx + 1
