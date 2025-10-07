import os
def prep():
    if(os.path.exists("confusion_matrix.typ")):
        os.remove("confusion_matrix.typ")
    if(os.path.exists("classification_tables.typ")):
        os.remove("classification_tables.typ")

def confusion_table(matrix,model):
    with open("confusion_matrix.typ","a") as f:
        name = model.replace(" ", "")
        f.write("#set align(center)\n")
        f.write(f"#let Table{name}confusionMatrix = figure(\n")
        f.write("  table(\n")
        f.write("    columns: 3,\n")
        f.write("    align:center,\n")
        f.write("    table.header(\n")
        f.write("      [],[Not Fire],[Fire]\n")
        f.write("    ),\n")
        f.write(f"    [Not Fire],[{matrix[0][0]}],[{matrix[0][1]}],\n")
        f.write(f"    [Fire],[{matrix[1][0]}],[{matrix[1][1]}],\n")
        f.write("  ),\n")
        f.write(f"  caption: [Confusion matrix for {model}.]\n")
        f.write(")\n")
        f.write("#set align(left)" + "\n")
        f.write("\n")

def classification_report_table(report, model):
    with open("classification_tables.typ","a") as f:
        name = model.replace(" ", "")
        f.write("#set align(center)\n")
        f.write(f"#let Table{name}classiffication = figure(\n")
        f.write("  table(\n")
        f.write("    columns: 5,\n")
        f.write("    align:center,\n")
        f.write("    table.header(\n")
        f.write("      [],[Precision],[Recall],[F1-Score],[Support]\n")
        f.write("    ),\n")
        f.write(f"    [Not Fire],[{report['Not Fire']['precision']:.2f}],[{report['Not Fire']['recall']:.2f}],[{report['Not Fire']['f1-score']:.2f}],[{report['Not Fire']['support']:.0f}],\n")
        f.write(f"    [Fire],[{report['Fire']['precision']:.2f}],[{report['Fire']['recall']:.2f}],[{report['Fire']['f1-score']:.2f}],[{report['Fire']['support']:.0f}],\n")
        f.write(f"    [accuracy],[],[],[{report['accuracy']:.2f}],[{report['Not Fire']['support']+report['Fire']['support']:.0f}],\n")
        f.write(f"    [macro avg],[{report['macro avg']['precision']:.2f}],[{report['macro avg']['recall']:.2f}],[{report['macro avg']['f1-score']:.2f}],[{report['macro avg']['support']:.0f}],\n")
        f.write(f"    [weighted avg],[{report['weighted avg']['precision']:.2f}],[{report['weighted avg']['recall']:.2f}],[{report['weighted avg']['f1-score']:.2f}],[{report['weighted avg']['support']:.0f}],\n")
        f.write("  ),\n")
        f.write(f"  caption: [Classification report for {model}.]\n")
        f.write(")\n")
        f.write("#set align(left)" + "\n")
        f.write("\n")