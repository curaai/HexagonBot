import tensorflow as tf
import numpy as np
import random
import time
import argparse

from game import Game
from model import DQN
from directkeys import PressKey, ReleaseKey, ENTER, L, R

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', default=True, type=bool, help="train or playing")
    parser.add_argument('-w', '--width', default=772, type=int, help="width for capturing rect")
    parser.add_argument('-he', '--height', default=500, type=int, help="height for capturing rect")
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float, help='Learning rate for training')
    parser.add_argument('-bs', '--batch_size', default=20, type=int, help="batch size")

    parser.add_argument('-e', '--max_episode', default=10000, type=int, help="max_episode for training")
    parser.add_argument('-ut', '--update_term', default=50, type=int, help="Update term for main to target")
    parser.add_argument('-tt', '--train_term', default=4, type=int, help="Train term ")
    parser.add_argument('-o', '--observe', default=100, type=int, help="Until to start network")
    parser.add_argument('-a', '--action_size', default=3, type=int, help="Size of action")
    parser.add_argument('-s', '--save_model', default='save/agent.cpkt', help="Saved trained model")

    args = parser.parse_args()


    def rgb2gray(rgb) -> float:
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    # train
    if args.train:
        with tf.Session() as sess:
            width, height = args.width, args.height
            game = Game((0, 25, width, height))
            dqn = DQN(sess, args.learning_rate, args.batch_size, 200, 200, args.action_size)

            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver()
            saver.restore(sess, args.save_model)

            epsilon = 1.0
            frame_count = 0
            rewards = []

            for i in range(4)[::-1]:
                print(i)
                time.sleep(1)
            PressKey(ENTER)
            time.sleep(0.05)
            ReleaseKey(ENTER)

            print("Learning Start !!!")
            for episode in range(args.max_episode):
                done = False
                total_reward = 0

                game.init_state()
                a = rgb2gray(game.state)
                dqn.init_state(rgb2gray(game.state))

                start = time.time()
                count = 0
                while not done:
                    if np.random.rand() < epsilon:
                        action = random.randrange(args.action_size)
                    else:
                        action = dqn.get_action()

                    if epsilon > args.observe:
                        epsilon -= 0.001

                    reward, done = game.step(action)
                    total_reward += reward

                    if start - time.time() > 1:
                        dqn.save_memory(action, reward, done, rgb2gray(game.state))
                        count += 1

                    if frame_count > args.observe and frame_count % args.train_term == 0 and count > 50:
                        dqn.train()

                    if frame_count % args.update_term == 0:
                        dqn.copy2target()

                    frame_count += 1
                PressKey(ENTER)
                time.sleep(0.1)
                ReleaseKey(ENTER)

                if episode % 10 == 0:
                    print("Iteration: {}, Score: {}".format(episode, total_reward))
                    rewards.append(total_reward)
                    total_reward = 0

                if episode % 100 == 0:
                    saver.save(sess, args.save_model)

