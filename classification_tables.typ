#set align(center)
#let Tabledecisiontreeclassiffication = figure(
  table(
    columns: 5,
    align:center,
    table.header(
      [],[Precision],[Recall],[F1-Score],[Support]
    ),
    [Not Fire],[1.00],[0.91],[0.95],[32],
    [Fire],[0.93],[1.00],[0.97],[42],
    [accuracy],[],[],[0.96],[74],
    [macro avg],[0.97],[0.95],[0.96],[74],
    [weighted avg],[0.96],[0.96],[0.96],[74],
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
    [Not Fire],[0.87],[0.81],[0.84],[32],
    [Fire],[0.86],[0.90],[0.88],[42],
    [accuracy],[],[],[0.86],[74],
    [macro avg],[0.87],[0.86],[0.86],[74],
    [weighted avg],[0.86],[0.86],[0.86],[74],
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
    [Not Fire],[1.00],[0.97],[0.98],[32],
    [Fire],[0.98],[1.00],[0.99],[42],
    [accuracy],[],[],[0.99],[74],
    [macro avg],[0.99],[0.98],[0.99],[74],
    [weighted avg],[0.99],[0.99],[0.99],[74],
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
    [Not Fire],[0.82],[0.88],[0.85],[32],
    [Fire],[0.90],[0.86],[0.88],[42],
    [accuracy],[],[],[0.86],[74],
    [macro avg],[0.86],[0.87],[0.86],[74],
    [weighted avg],[0.87],[0.86],[0.87],[74],
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
    [Not Fire],[0.97],[0.94],[0.95],[32],
    [Fire],[0.95],[0.98],[0.96],[42],
    [accuracy],[],[],[0.96],[74],
    [macro avg],[0.96],[0.96],[0.96],[74],
    [weighted avg],[0.96],[0.96],[0.96],[74],
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
    [Not Fire],[1.00],[0.94],[0.97],[32],
    [Fire],[0.95],[1.00],[0.98],[42],
    [accuracy],[],[],[0.97],[74],
    [macro avg],[0.98],[0.97],[0.97],[74],
    [weighted avg],[0.97],[0.97],[0.97],[74],
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
    [Not Fire],[0.82],[0.84],[0.83],[32],
    [Fire],[0.88],[0.86],[0.87],[42],
    [accuracy],[],[],[0.85],[74],
    [macro avg],[0.85],[0.85],[0.85],[74],
    [weighted avg],[0.85],[0.85],[0.85],[74],
  ),
  caption: [Classification report for kNN.]
)
#set align(left)

