import argparse

import matplotlib.pyplot as plt
import pandas as pd


def plot_scores(filename, title=None, skip=0):
    scores = pd.read_csv(filename, delimiter=',', skiprows=range(1, skip+1))
    for col in ['train_loss', 'test_loss']:
        plt.plot(scores['learn_t'], scores[col], label=col)
    if title is not None:
        plt.title(title)
    plt.xlabel('learn_t')
    plt.ylabel('loss')
    plt.legend(loc='best')
    fig_fname = filename + '.png'
    plt.savefig(fig_fname)
    print('Saved a figure as {}'.format(fig_fname))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scores', type=str, help='specify path of scores.txt')
    parser.add_argument('--title', type=str, default=None)
    parser.add_argument('--skip', type=int, default=0)
    args = parser.parse_args()
    plot_scores(args.scores, args.title, args.skip)
