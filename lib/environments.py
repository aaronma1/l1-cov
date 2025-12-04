

def mountaincar_bounds():

    s_low = [-1.2, -0.07]
    s_high = [0.6, 0.07]
    a_low = [-1.0]
    a_high = [1.0]

    return s_low, s_high, a_low, a_high


def pendulum_bounds():
    s_low = [-1.0, -1.0, -8.0]
    s_high = [1.0, 1.0, 8.0]
    
    a_low =[-1.0]
    a_high = [1.0]
    return s_low, s_high, a_low, a_high

def cartpole_bounds():
    s_low = [-2.5, -3.5, -0.3, -4.0]
    s_high=[2.5,3.5, 0.3, 4.0]

    return s_low, s_high

def acrobot_bounds():
    s_low = [-1.0,-1.0, -1.0, -1.0, -13, -28.5]
    s_high = [1.0, 1.0, 1.0, 1.0, 13, 28.5]

    return s_low, s_high



