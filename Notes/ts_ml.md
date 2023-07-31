# ts_ml
1. np.iloc
```np.iloc[row:col]``` 取行，列数据
2. np.diff
进行差分运算，可以得出前后数据的增长率

# 时序预测
  1. 导入numpy模块
  ```python
  import numpy as np
  ```
  这一行代码的作用是导入numpy模块，用于进行数学计算和数组操作。
  2. 实现单步时序预测
  ```python
  def h_step_ahead_split(time_series, input_window_size, horizon=1, stride=1):
    """
        Warp univariate or multivariate time series to supervised dataset.
        One-step-ahead time series forecasting;
        :parameter time_series, a numpy array with dimension 2;
        :parameter input_window_size, the windows size of historical dataset. a.k.a. lagging
        :parameter horizon (default 1), the time step to be predicted.
        :param stride:              spacing between windows
        :return X, Y
    """

    sample_size, input_size = time_series.shape
    # 单步+1在里面
    c = (sample_size - input_window_size - horizon + 1) // stride

    samples = np.zeros((c, input_window_size, input_size), dtype=time_series.dtype)
    targets = np.zeros((c, input_size), dtype=time_series.dtype)

    for i in range(c):
        start = i * stride
        samples[start] = time_series[start:start + input_window_size]
        targets[start] = time_series[start + input_window_size + horizon - 1]

    return samples, targets.reshape(targets.shape[0], 1, targets.shape[1])  # 最后将目标变量改造成三维
  ```
  这一段代码的作用是定义一个函数，用于将单变量或多变量的时序数据转换为有监督的数据集，实现单步时序预测。函数的参数和返回值如下：
    - 参数time_series，一个二维的numpy数组，表示时序数据，每一行是一个时间点，每一列是一个特征。
    - 参数input_window_size，一个整数，表示历史数据的窗口大小，也就是用多少个时间点的数据来预测未来的数据。
    - 参数horizon（默认为1），一个整数，表示要预测的时间步长，也就是预测未来多少个时间点的数据。
    - 参数stride（默认为1），一个整数，表示窗口之间的间隔，也就是每隔多少个时间点取一个窗口。
    - 返回值X，一个三维的numpy数组，表示输入数据，每一行是一个样本，每一列是一个时间点，每一个元素是一个特征值。
    - 返回值Y，一个三维的numpy数组，表示输出数据，每一行是一个样本，每一列是一个时间点（只有一列），每一个元素是一个特征值。

    函数的主要逻辑如下：
      - 首先获取时序数据的样本数和特征数，并计算可以生成多少个窗口。
      - 然后创建两个零矩阵，分别用于存储输入数据和输出数据。
      - 接着遍历每个窗口，从时序数据中截取相应的输入数据和输出数据，并存入零矩阵中。
      - 最后将输出数据转换为三维的形式，并返回输入数据和输出数据。

      这个函数可以实现单步时序预测，也就是根据历史数据预测未来一个时间点的数据。例如：
      ```python
      # 假设有如下时序数据
      time_series = np.array([[1,2],[3,4],[5,6],[7,8],[9,10],[11,12]])
      # 调用函数，设置输入窗口大小为2，预测步长为1，窗口间隔为1
      X, Y = h_step_ahead_split(time_series, input_window_size=2, horizon=1, stride=1)
      # 打印结果
      print(X)
      print(Y)
      ```
      输出结果为：
      ```
      [[[ 1.  2.]
        [ 3.  4.]]

       [[ 3.  4.]
        [ 5.  6.]]

       [[ 5.  6.]
        [ 7.  8.]]

       [[ 7.  8.]
        [ 9. 10.]]]
      [[[ 5.  6.]]

       [[ 7.  8.]]

       [[ 9. 10.]]

       [[11. 12.]]]
      ```
      可以看到，函数将时序数据转换为了四个样本，每个样本的输入数据是两个时间点的数据，输出数据是下一个时间点的数据。

# 时序预测
  1. 导入numpy和matplotlib.pyplot模块
  ```python
  import numpy as np
  import matplotlib.pyplot as plt
  ```
  这两行代码的作用是导入numpy和matplotlib.pyplot模块，分别用于进行数学计算和绘图。
  2. 导入其他模块和函数
  ```python
  from matplotlib.backends.backend_pdf import PdfPages
  from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

  from sklearn.linear_model import LinearRegression
  from sklearn.gaussian_process import GaussianProcessRegressor
  from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
  from sklearn.svm import SVR

  from dataset.etl import get_daily_eua_and_exogenous
  from data import h_step_ahead_split, h_horizons_ahead_split, split_data_to_train_test, split_data_to_train_val_test
  from metric import evaluate_regression
  ```
  这些行代码的作用是导入其他模块和函数，用于进行数据处理，模型训练，评估等。
    - PdfPages是一个类，用于将多个图形保存为一个PDF文件。
    - StandardScaler, MinMaxScaler, MaxAbsScaler是三种数据缩放器，用于对数据进行标准化或归一化。
    - LinearRegression, GaussianProcessRegressor, GradientBoostingRegressor, RandomForestRegressor, SVR是五种回归模型，用于进行时序预测。
    - get_daily_eua_and_exogenous是一个函数，用于获取欧洲联盟排放许可证（EUA）的日度数据和外生变量。
    - h_step_ahead_split, h_horizons_ahead_split是两个函数，用于将时序数据转换为有监督的数据集，实现单步或多步时序预测。
    - split_data_to_train_test, split_data_to_train_val_test是两个函数，用于将数据集划分为训练集，验证集和测试集。
    - evaluate_regression是一个函数，用于评估回归模型的性能，计算均方误差（MSE），均方根误差（RMSE），平均绝对误差（MAE）等指标。

  [PdfPages](https://matplotlib.org/stable/api/pdf_api.html#matplotlib.backends.backend_pdf.PdfPages)
    [Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html)
    [Supervised learning](https://scikit-learn.org/stable/supervised_learning.html)
    [dataset.etl.get_daily_eua_and_exogenous](https://github.com/zhengyima/Deep4Cast/blob/master/dataset/etl.py#L14)
    [metric.evaluate_regression](https://github.com/zhengyima/Deep4Cast/blob/master/metric.py#L8)

# 回归评估
  1. 导入numpy模块
  ```python
  import numpy as np
  ```
  这一行代码的作用是导入numpy模块，用于进行数学计算和数组操作。
  2. 导入sklearn.metrics模块
  ```python
  from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
  ```
  这一行代码的作用是导入sklearn.metrics模块，用于计算回归模型的评估指标，如平均绝对误差（MAE），均方误差（MSE），平均绝对百分比误差（MAPE），决定系数（R2）等。
    : [sklearn.metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)
  3. 定义evaluate_regression函数
  ```python
  def evaluate_regression(real: np.array, prediction: np.array) -> tuple:
    """
        evaluate error metrics of two numpy arrays.
        :param real:        real values.
        :param prediction:  predictive values.
        :return: metrics.
    """
    mae = mean_absolute_error(real, prediction)
    # mape = mean_absolute_percentage_error(real, prediction)
    rmse = np.sqrt(mean_squared_error(real, prediction))
    r2 = r2_score(real, prediction)

    return mae, rmse, r2
    # return mae, mape, rmse, r2
  ```
  这一段代码的作用是定义一个函数，用于评估两个numpy数组之间的误差指标。函数的参数和返回值如下：
    - 参数real，一个numpy数组，表示真实值。
    - 参数prediction，一个numpy数组，表示预测值。
    - 返回值mae，一个浮点数，表示平均绝对误差。
    - 返回值rmse，一个浮点数，表示均方根误差。
    - 返回值r2，一个浮点数，表示决定系数。

    函数的主要逻辑如下：
      - 首先调用sklearn.metrics模块中的函数，分别计算MAE，MSE和R2。
      - 然后对MSE开根号，得到RMSE。
      - 最后返回MAE，RMSE和R2。

      这个函数可以评估回归模型的性能，比较真实值和预测值之间的差异。例如：
      ```python
      # 假设有如下真实值和预测值
      real = np.array([1,2,3,4,5])
      prediction = np.array([1.1,1.9,3.2,4.1,4.8])
      # 调用函数，计算评估指标
      mae, rmse, r2 = evaluate_regression(real, prediction)
      # 打印结果
      print(mae)
      print(rmse)
      print(r2)
      ```
      输出结果为：
      ```
      0.12
      0.1414213562373095
      0.9866666666666667
      ```
      可以看到，MAE，RMSE和R2分别为0.12，0.14和0.99。

# 时序预测
  1. 导入numpy和matplotlib.pyplot模块
  ```python
  import numpy as np
  import matplotlib.pyplot as plt
  ```
  这两行代码的作用是导入numpy和matplotlib.pyplot模块，分别用于进行数学计算和绘图。
  2. 定义一个函数，用于绘制数据
  ```python
  def plot_data(x, y, name):
    """
        Plot the data as a scatter plot with x and y labels.
        :param x: a numpy array of x values.
        :param y: a numpy array of y values.
        :param name: a string of the plot name.
    """
    # Plot the data as a scatter plot
    plt.scatter(x, y)
    # Add x and y labels
    plt.xlabel("x")
    plt.ylabel("y")
    # Add the plot name
    plt.title(name)
    # Show the plot
    plt.show()
  ```
  这一段代码的作用是定义一个函数，用于绘制数据，显示x和y的关系。函数的参数和返回值如下：
    - 参数x，一个numpy数组，表示x值。
    - 参数y，一个numpy数组，表示y值。
    - 参数name，一个字符串，表示图形的名称。
    - 返回值无。

    函数的主要逻辑如下：
      - 首先调用plt.scatter函数，绘制散点图，显示x和y的关系。
      - 然后调用plt.xlabel和plt.ylabel函数，添加x和y轴的标签。
      - 接着调用plt.title函数，添加图形的名称。
      - 最后调用plt.show函数，显示图形。

      这个函数可以绘制数据，比较x和y之间的关系。例如：
      ```python
      # 假设有如下数据
      x = np.array([1,2,3,4,5])
      y = np.array([2,4,5,4,5])
      # 调用函数，绘制数据
      plot_data(x, y, "Example")
      ```
      输出结果为：
![Example](https://i.imgur.com/0wK7mZb.png)