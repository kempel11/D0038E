#set align(center)
#let TabledecisiontreeconfusionMatrix = figure(
  table(
    columns: 3,
    align:center,
    table.header(
      [],[Not Fire],[Fire]
    ),
    [Not Fire],[29],[3],
    [Fire],[0],[42],
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
      [],[Not Fire],[Fire]
    ),
    [Not Fire],[31],[1],
    [Fire],[2],[40],
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
      [],[Not Fire],[Fire]
    ),
    [Not Fire],[31],[1],
    [Fire],[0],[42],
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
      [],[Not Fire],[Fire]
    ),
    [Not Fire],[29],[3],
    [Fire],[2],[40],
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
      [],[Not Fire],[Fire]
    ),
    [Not Fire],[30],[2],
    [Fire],[1],[41],
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
      [],[Not Fire],[Fire]
    ),
    [Not Fire],[31],[1],
    [Fire],[0],[42],
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
      [],[Not Fire],[Fire]
    ),
    [Not Fire],[26],[6],
    [Fire],[4],[38],
  ),
  caption: [Confusion matrix for kNN.]
)
#set align(left)

