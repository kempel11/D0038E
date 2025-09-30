import seaborn as sns
import matplotlib.pyplot as plt

def display_confusion_matrix(matrix, name):
    """
    :param matrix: confusion_matrix to display
    :param name: file will be named name + "_confusion_matrix.png"
    """
    plot = sns.heatmap(matrix, cbar=False,  annot=True, cmap=sns.light_palette("#01385e",as_cmap=True))
    plot.set_xticklabels(labels=["Fire","Not Fire"])
    plot.set_xlabel("Prediction")
    plot.xaxis.tick_top()
    plot.xaxis.set_label_position('top')
    plot.set_yticklabels(labels=["Fire","Not Fire"])
    plot.set_ylabel("Actual")

    plt.savefig(f"figure/confusion_matrix/{name}_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()