#set align(center)
#let Tabledecisiontreeclassiffication = figure(
  table(
    columns: 5,
    align:center,
    table.header(
      [],[Precision],[Recall],[F1-Score],[Support]
    ),
    [Not Fire],[0.94],[1.00],[0.97],[16],
    [Fire],[1.00],[0.95],[0.98],[21],
    [accuracy],[],[],[0.97],[37],
    [macro avg],[0.97],[0.98],[0.97],[37],
    [weighted avg],[0.97],[0.97],[0.97],[37],
  ),
  caption: [Classification report for decision tree.]
)
#set align(left)

#set align(center)
#let TableSVMclassiffication = figure(
  table(
    columns: 5,
    align:center,
    table.header(
      [],[Precision],[Recall],[F1-Score],[Support]
    ),
    [Not Fire],[0.88],[0.94],[0.91],[16],
    [Fire],[0.95],[0.90],[0.93],[21],
    [accuracy],[],[],[0.92],[37],
    [macro avg],[0.92],[0.92],[0.92],[37],
    [weighted avg],[0.92],[0.92],[0.92],[37],
  ),
  caption: [Classification report for SVM.]
)
#set align(left)

#set align(center)
#let TableRandomForestclassiffication = figure(
  table(
    columns: 5,
    align:center,
    table.header(
      [],[Precision],[Recall],[F1-Score],[Support]
    ),
    [Not Fire],[1.00],[0.94],[0.97],[16],
    [Fire],[0.95],[1.00],[0.98],[21],
    [accuracy],[],[],[0.97],[37],
    [macro avg],[0.98],[0.97],[0.97],[37],
    [weighted avg],[0.97],[0.97],[0.97],[37],
  ),
  caption: [Classification report for Random Forest.]
)
#set align(left)

#set align(center)
#let TableMLPclassiffication = figure(
  table(
    columns: 5,
    align:center,
    table.header(
      [],[Precision],[Recall],[F1-Score],[Support]
    ),
    [Not Fire],[0.94],[0.94],[0.94],[16],
    [Fire],[0.95],[0.95],[0.95],[21],
    [accuracy],[],[],[0.95],[37],
    [macro avg],[0.94],[0.94],[0.94],[37],
    [weighted avg],[0.95],[0.95],[0.95],[37],
  ),
  caption: [Classification report for MLP.]
)
#set align(left)

#set align(center)
#let TableAdaBoostclassiffication = figure(
  table(
    columns: 5,
    align:center,
    table.header(
      [],[Precision],[Recall],[F1-Score],[Support]
    ),
    [Not Fire],[0.94],[1.00],[0.97],[16],
    [Fire],[1.00],[0.95],[0.98],[21],
    [accuracy],[],[],[0.97],[37],
    [macro avg],[0.97],[0.98],[0.97],[37],
    [weighted avg],[0.97],[0.97],[0.97],[37],
  ),
  caption: [Classification report for AdaBoost.]
)
#set align(left)

#set align(center)
#let TableGradientBoostclassiffication = figure(
  table(
    columns: 5,
    align:center,
    table.header(
      [],[Precision],[Recall],[F1-Score],[Support]
    ),
    [Not Fire],[0.94],[0.94],[0.94],[16],
    [Fire],[0.95],[0.95],[0.95],[21],
    [accuracy],[],[],[0.95],[37],
    [macro avg],[0.94],[0.94],[0.94],[37],
    [weighted avg],[0.95],[0.95],[0.95],[37],
  ),
  caption: [Classification report for Gradient Boost.]
)
#set align(left)

#set align(center)
#let TablekNNclassiffication = figure(
  table(
    columns: 5,
    align:center,
    table.header(
      [],[Precision],[Recall],[F1-Score],[Support]
    ),
    [Not Fire],[0.88],[0.88],[0.88],[16],
    [Fire],[0.90],[0.90],[0.90],[21],
    [accuracy],[],[],[0.89],[37],
    [macro avg],[0.89],[0.89],[0.89],[37],
    [weighted avg],[0.89],[0.89],[0.89],[37],
  ),
  caption: [Classification report for kNN.]
)
#set align(left)

