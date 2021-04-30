import matplotlib.pyplot as plt
import numpy as np


def plot_data(y_testing_data, title, src_path):
    unique, counts = np.unique(y_testing_data, return_counts=True)
    total_properties = y_testing_data.size

    labels = ["Nothing in hand", "One pair", "Two pairs", "Three of a kind", "Straight", "Flush", "Full house",
              "Four of a kind", "Straight flush", "Royal flush"]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, counts, width, label='Amount of specific poker hands')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.yscale('log')
    ax.set_xlabel('Poker hands', fontsize=12)
    ax.set_ylabel('Number of layouts', fontsize=12)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)

    fig.set_figheight(6)
    fig.set_figwidth(17)
    fig.tight_layout()
    fig.savefig(src_path, dpi=400)

    plt.show()


def plot_missing_data(x_training_data, x_testing_data, title, src_path):
    count_training_data = x_training_data.shape[0]
    count_testing_data = x_testing_data.shape[0]

    labels = ["All combinations", "Training", "Testing"]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, [52 * 51 * 50 * 49 * 48, count_testing_data, count_training_data], width,
                    label='Amount of poker hands in datasets')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.yscale('log')
    ax.set_xlabel('Amount of poker hands layouts', fontsize=12)
    ax.set_ylabel('Unique layouts', fontsize=12)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)

    fig.set_figheight(6)
    fig.set_figwidth(17)
    fig.tight_layout()
    fig.savefig(src_path, dpi=400)

    plt.show()
