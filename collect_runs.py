from tilecoding import MountainCarTileCoder
import gymnasium as gym











if __name__ == "__main__":



    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")

    state_tc = MountainCarTileCoder(iht_size=4096, num_tiles=32, num_tilings=1)
    agent_tc = MountainCarTileCoder(iht_size=4096, num_tiles=16, num_tilings=16)


    print(state_tc.tile_state([-1.2, -0.7]))


    # rod_cycle(env, state_tc)

