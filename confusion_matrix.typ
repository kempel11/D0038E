#set align(center)
#let TabledecisiontreeconfusionMatrix = figure(
  table(
    columns: 3,
    align:center,
    table.header(
      [],[Fire],[Not Fire]
    ),
    [Fire],[16],[0],
    [Not Fire],[1],[20],
  ),
  caption: [Confusion matrix for decision tree.]
)
#set align(left)

#set align(center)
#let TableSVMconfusionMatrix = figure(
  table(
    columns: 3,
    align:center,
    table.header(
      [],[Fire],[Not Fire]
    ),
    [Fire],[15],[1],
    [Not Fire],[2],[19],
  ),
  caption: [Confusion matrix for SVM.]
)
#set align(left)

#set align(center)
#let TableRandomForestconfusionMatrix = figure(
  table(
    columns: 3,
    align:center,
    table.header(
      [],[Fire],[Not Fire]
    ),
    [Fire],[15],[1],
    [Not Fire],[0],[21],
  ),
  caption: [Confusion matrix for Random Forest.]
)
#set align(left)

#set align(center)
#let TableMLPconfusionMatrix = figure(
  table(
    columns: 3,
    align:center,
    table.header(
      [],[Fire],[Not Fire]
    ),
    [Fire],[15],[1],
    [Not Fire],[1],[20],
  ),
  caption: [Confusion matrix for MLP.]
)
#set align(left)

#set align(center)
#let TableAdaBoostconfusionMatrix = figure(
  table(
    columns: 3,
    align:center,
    table.header(
      [],[Fire],[Not Fire]
    ),
    [Fire],[16],[0],
    [Not Fire],[1],[20],
  ),
  caption: [Confusion matrix for AdaBoost.]
)
#set align(left)

#set align(center)
#let TableGradientBoostconfusionMatrix = figure(
  table(
    columns: 3,
    align:center,
    table.header(
      [],[Fire],[Not Fire]
    ),
    [Fire],[15],[1],
    [Not Fire],[1],[20],
  ),
  caption: [Confusion matrix for Gradient Boost.]
)
#set align(left)

#set align(center)
#let TablekNNconfusionMatrix = figure(
  table(
    columns: 3,
    align:center,
    table.header(
      [],[Fire],[Not Fire]
    ),
    [Fire],[14],[2],
    [Not Fire],[2],[19],
  ),
  caption: [Confusion matrix for kNN.]
)
#set align(left)

