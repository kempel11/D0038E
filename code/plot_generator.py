import seaborn as sns
import matplotlib.pyplot as plt

def display_confusion_matrix(matrix, name):
    """
    :param matrix: confusion_matrix to display
    :param name: file will be named name + "_confusion_matrix.png"
    """
    plot = sns.heatmap(matrix, cbar=False,  annot=True, cmap=sns.light_palette("#01385e",as_cmap=True))
    plot.set_xticklabels(labels=["Not Fire", "Fire"])
    plot.set_xlabel("Prediction")
    plot.xaxis.tick_top()
    plot.xaxis.set_label_position('top')
    plot.set_yticklabels(labels=["Not Fire","Fire"])
    plot.set_ylabel("Actual")

    plt.savefig(f"figure/confusion_matrix/{name}_confusion_matrix.png", dpi=300, bbox_inches="tight")
    #plt.show()
    plt.close()

def accuracy_comparison(reports,names):
    accuracy = []
    for i in range(len(reports)):
        accuracy.append(reports[i]['accuracy'])
    plot = plt.bar(names, accuracy, color="#01385e")
    plt.bar_label(plot, fmt='{:,.3f}')
    plt.xticks(rotation=45)
    plt.title("Accuracy")
    plt.savefig("figure/comparison/accuracy_bar_chart.png", dpi=300, bbox_inches="tight")
    #plt.show()
    plt.close()

def f1_score_comparison(reports,names):
    f1 = []
    for i in range(len(reports)):
        f1.append(reports[i]['macro avg']['f1-score'])

    plot = plt.bar(names, f1, color="#01385e")
    plt.bar_label(plot, fmt='{:,.3f}')
    plt.xticks(rotation=45)
    plt.title("F1-Score") 
    plt.savefig("figure/comparison/f1_score_bar_chart.png", dpi=300, bbox_inches="tight")
    #plt.show()
    plt.close()