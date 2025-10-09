from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def display_confusion_matrix(matrix, name):
    """
    :param matrix: confusion_matrix to display
    :param name: file will be named name + "_confusion_matrix.png"
    """
    plot = sns.heatmap(matrix, cbar=False,  annot=True, cmap=sns.color_palette("blend:#FFCC11,#992222", as_cmap=True))
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
    plot = plt.bar(names, accuracy, color="#FFCC11")
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

    plot = plt.bar(names, f1, color="#FFCC11")
    plt.bar_label(plot, fmt='{:,.3f}')
    plt.xticks(rotation=45)
    plt.title("F1-Score") 
    plt.savefig("figure/comparison/f1_score_bar_chart.png", dpi=300, bbox_inches="tight")
    #plt.show()
    plt.close()

def recall_score_comparison(report, names):
    x = np.arange(len(names))*2
    not_fire_recall = []
    fire_recall = []
    for i in range(len(names)):
        not_fire_recall.append(report[i]['Not Fire']['recall'])
        fire_recall.append(report[i]['Fire']['recall'])
    
    width=0.9
    plot1 = plt.bar(x-width/2,not_fire_recall, width, color='#FFCC11',label='Not Fire')
    plot2 = plt.bar(x+width/2,fire_recall, width, color='#992222', label='Fire')
    plt.xticks(x, names, rotation=45)
    plt.legend(loc='upper left', ncols=2)
    plt.bar_label(plot1, fmt='{:,.3f}', fontsize=8)
    plt.bar_label(plot2, fmt='{:,.3f}', fontsize=8)
    plt.title("Recall")
    plt.ylim(top=1.2)
    plt.tight_layout()
    plt.savefig("figure/comparison/recall_score_bar_chart.png", dpi=300, bbox_inches="tight")
    #plt.show()