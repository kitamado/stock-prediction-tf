# stock-prediction-tf

Use DNN, CNN, RNN to predict stock price.

- CNN

  - 将股票涨幅看作分类问题

    - | Rise       | Rise label |
      | ---------- | ---------- |
      | %6 ~%10.1  | 3          |
      | %3 ~ %6    | 2          |
      | 0 ~ %3     | 1          |
      | -%10.1 ~ 0 | 0          |

  - ACC/LOSS

    - ![cnn_al](https://github.com/kitamado/stock-prediction-tf/blob/main/CNN_Rice_Label_ACC_LOSS.png)

  - Prediction

    - ![cnn_pred](https://github.com/kitamado/stock-prediction-tf/blob/main/CNN_Rice_Label_Pridiction.png)

- RNN

  - 预测收盘价
  - LOSS
    - ![RNN_loss](https://github.com/kitamado/stock-prediction-tf/blob/main/RNN_Close_Price_Loss.png)
  - Prediction
    - ![Rnn_pred](https://github.com/kitamado/stock-prediction-tf/blob/main/RNN_Close_Price_Pridiction.png)

  

