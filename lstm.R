###########################
# 1. Cài đặt và nạp các gói cần thiết
###########################
packages <- c("readxl", "xts", "zoo", "tseries", "rugarch", "forecast", "FinTS")
installed <- rownames(installed.packages())
for (pkg in packages) {
  if (!(pkg %in% installed)) install.packages(pkg)
  library(pkg, character.only = TRUE)
}
#use_python("C:/Users/quanp/AppData/Local/Programs/Python/Python312/python.exe", required = TRUE)

# Nếu bạn đã cài Python và tensorflow từ trước, có thể khai báo rõ:
# library(reticulate)
# use_python("C:/Users/quanp/AppData/Local/Programs/Python/Python312/python.exe", required = TRUE)
#install_keras(tensorflow = "2.16.1")
###########################
# 2. Đọc và xử lý dữ liệu
###########################
data <- read_excel("data_vnindex1.xlsx")
names(data) <- tolower(names(data))
data$date <- as.Date(data$date)
data <- data[order(data$date), ]
data_xts <- xts(data[, setdiff(names(data), "date")], order.by = data$date)

###########################
# 3. Chia dữ liệu thành tập huấn luyện và kiểm tra
###########################
training_data <- data_xts["/2024-03-31"]
test_data     <- data_xts["2024-04-01/"]

# Kiểm định tính dừng chuỗi log-return
log_return_train <- diff(log(training_data$vnindex_close))
log_return_train <- na.omit(log_return_train)

cat("\n--- Kiểm định tính dừng ---\n")
cat("ADF test:\n")
print(adf.test(log_return_train))
cat("\nKPSS test:\n")
print(kpss.test(log_return_train))

###########################
# 4. Ước lượng mô hình ARIMA để lấy thông số
###########################
fit_arima <- auto.arima(log_return_train)
print(summary(fit_arima))
arima_order <- arimaorder(fit_arima)

###########################
# 5. Định nghĩa các mô hình GARCH biến thể
###########################
spec_arima_garch <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1,1)),
  mean.model = list(armaOrder = c(arima_order[1], arima_order[3]), include.mean = TRUE),
  distribution.model = "std"
)

spec_arima_egarch <- ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1)),
  mean.model = list(armaOrder = c(arima_order[1], arima_order[3]), include.mean = TRUE),
  distribution.model = "std"
)

spec_arima_gjr <- ugarchspec(
  variance.model = list(model = "gjrGARCH", garchOrder = c(1,1)),
  mean.model = list(armaOrder = c(arima_order[1], arima_order[3]), include.mean = TRUE),
  distribution.model = "std"
)

spec_garch <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1,1)),
  mean.model = list(armaOrder = c(0, 0), include.mean = TRUE),
  distribution.model = "std"
)

###########################
# 6. Dự báo rolling từng bước cho các mô hình GARCH
###########################
forecast_horizon <- nrow(test_data)
last_train_value <- as.numeric(tail(training_data$vnindex_close, 1))
actual <- as.numeric(test_data$vnindex_close)
forecast_dates <- index(test_data)[1:forecast_horizon]

models <- list(
  ARIMA = fit_arima,
  GARCH = spec_garch,
  ARIMA_GARCH = spec_arima_garch,
  ARIMA_EGARCH = spec_arima_egarch,
  ARIMA_GJR_GARCH = spec_arima_gjr
)

results <- list()

# ARIMA dự báo riêng
forecast_arima <- forecast(models$ARIMA, h = forecast_horizon)
results[["ARIMA"]] <- exp(cumsum(as.numeric(forecast_arima$mean))) * last_train_value

# GARCH-type models dự báo rolling
for (name in names(models)[-1]) {
  cat("Running model:", name, "\n")
  roll <- ugarchroll(
    models[[name]],
    data = log_return_train,
    n.ahead = 1,
    forecast.length = forecast_horizon,
    refit.every = 5,
    refit.window = "moving",
    solver = "hybrid"
  )
  mu <- as.numeric(roll@forecast$density[,"Mu"])
  mu_drift_adj <- mu - mean(mu)
  level <- exp(cumsum(mu_drift_adj)) * last_train_value
  results[[name]] <- level
}

###########################
# 7. Xây và huấn luyện mô hình LSTM (nếu khả dụng)
###########################
price_train <- as.numeric(training_data$vnindex_close)
price_test <- as.numeric(test_data$vnindex_close)
price_scaled <- scale(price_train)
n_samples <- as.integer(length(price_scaled) - 10)
X <- array(NA, dim = c(n_samples, 10L, 1L))
y <- array(NA, dim = c(n_samples))
for (i in 1:n_samples) {
  X[i, , 1] <- as.numeric(price_scaled[i:(i+9)])
  y[i] <- as.numeric(price_scaled[i+10])
}

if (requireNamespace("keras", quietly = TRUE) && keras::is_keras_available()) {
  library(keras)
  
  # Định nghĩa input layer
  input <- layer_input(shape = c(10, 1))
  
  # Xây dựng các layer kế tiếp bằng cách nối với input
  output <- input %>%
    layer_lstm(units = 50, return_sequences = FALSE) %>%
    layer_dense(units = 1)
  
  # Tạo mô hình từ input và output
  model_lstm <- keras_model(inputs = input, outputs = output)
  
  model_lstm$compile(
    loss = "mean_squared_error",
    optimizer = optimizer_adam()
  )
  
  history <- model_lstm$fit(
    x = X,
    y = y,
    epochs = 50,
    batch_size = 16,
    verbose = 0
  )
  
  # Dự báo
  last_sequence <- tail(price_scaled, 10)
  preds_scaled <- numeric(forecast_horizon)
  forecast_horizon <- as.integer(nrow(test_data))
  for (i in 1:forecast_horizon) {
    input_seq <- array(last_sequence, dim = c(1, 10, 1))
    pred <- model_lstm %>% predict(input_seq)
    preds_scaled[i] <- pred
    last_sequence <- c(last_sequence[-1], pred)
  }
  
  preds <- preds_scaled * attr(price_scaled, 'scaled:scale') + attr(price_scaled, 'scaled:center')
  results[["LSTM"]] <- preds
}

#############################
# 7.1. Triển khai mô hình LSTM-LWR
#############################

# Hàm DLWR sử dụng LWR thông qua loess()
DLWR_decompose <- function(R, spans = c(0.3, 0.3, 0.3), degree = 2) {
  # R: vector dữ liệu gốc
  # spans: vector chứa các giá trị span cho 3 lần tách
  # degree: bậc polynomial cho loess (1 hoặc 2)
  n <- length(R)
  t <- seq_len(n)
  
  # Lần tách 1: R = f0 + d0
  fit_f0 <- loess(R ~ t, span = spans[1], degree = degree)
  f0 <- predict(fit_f0, newdata = data.frame(t = t))
  d0 <- R - f0
  
  # Lần tách 2: d0 = f1 + d1
  fit_f1 <- loess(d0 ~ t, span = spans[2], degree = degree)
  f1 <- predict(fit_f1, newdata = data.frame(t = t))
  d1 <- d0 - f1
  
  # Lần tách 3: d1 = f2 + d2
  fit_f2 <- loess(d1 ~ t, span = spans[3], degree = degree)
  f2 <- predict(fit_f2, newdata = data.frame(t = t))
  d2 <- d1 - f2
  
  # Trả về các thành phần sao cho R ≈ f0 + f1 + f2 + d2
  return(list(f0 = f0, f1 = f1, f2 = f2, d2 = d2))
}

# Sử dụng cột giá trị đóng cửa của VN-Index từ tập huấn luyện (training_data)
raw_series <- as.numeric(training_data$vnindex_close)
decomp <- DLWR_decompose(raw_series, spans = c(0.3, 0.3, 0.3), degree = 2)

# (Tùy chọn) Vẽ biểu đồ các thành phần DLWR để kiểm tra
plot(raw_series, type = "l", col = "black", lwd = 2, 
     main = "Phân tách DLWR", xlab = "Thời gian", ylab = "Giá trị")
lines(decomp$f0, col = "blue", lwd = 2)
lines(decomp$f1, col = "green", lwd = 2)
lines(decomp$f2, col = "red", lwd = 2)
lines(decomp$d2, col = "purple", lwd = 2)
legend("topleft", legend = c("Raw", "f0", "f1", "f2", "d2"),
       col = c("black", "blue", "green", "red", "purple"), lty = 1, lwd = 2)

#############################
# 7.2. Huấn luyện LSTM cho thành phần f2 (thành phần xu hướng thứ ba)
#############################
# Chọn thành phần f2 để dự báo
f2 <- decomp$f2

# Chuẩn hóa f2 về khoảng [0, 1] (MinMaxScaler)
f2_min <- min(f2, na.rm = TRUE)
f2_max <- max(f2, na.rm = TRUE)
f2_scaled <- (f2 - f2_min) / (f2_max - f2_min)

# Tạo dữ liệu dạng chuỗi cho LSTM: sử dụng 'lag' bước làm input
lag <- 10  # số bước thời gian (window) cho input
num_samples <- length(f2_scaled) - lag
X <- array(NA, dim = c(num_samples, lag, 1))
y <- array(NA, dim = c(num_samples))
for (i in 1:num_samples) {
  X[i, , 1] <- f2_scaled[i:(i + lag - 1)]
  y[i] <- f2_scaled[i + lag]
}

# Kiểm tra gói keras đã được cài đặt và khả dụng chưa
if(requireNamespace("keras", quietly = TRUE) && keras::is_keras_available()){
  library(keras)
  
  # Xây dựng mô hình LSTM
  model_lstm_lwr <- keras_model_sequential() %>%
    layer_lstm(units = 50, input_shape = c(lag, 1)) %>%
    layer_dense(units = 1)
  
  model_lstm_lwr %>% compile(
    loss = "mean_squared_error",
    optimizer = "adam"
  )
  
  # Huấn luyện mô hình LSTM cho f2
  model_lstm_lwr %>% fit(X, y, epochs = 50, batch_size = 16, verbose = 1)
  
  #############################
  # 7.3. Dự báo tương lai cho f2 bằng mô hình LSTM
  #############################
  forecast_horizon <- nrow(test_data)  # số bước dự báo bằng độ dài của tập kiểm tra
  last_seq <- tail(f2_scaled, lag)
  preds_scaled <- numeric(forecast_horizon)
  
  for(i in 1:forecast_horizon){
    input_seq <- array(last_seq, dim = c(1, lag, 1))
    pred <- model_lstm_lwr %>% predict(input_seq)
    preds_scaled[i] <- pred
    # Cập nhật chuỗi: loại bỏ giá trị đầu tiên và thêm dự báo mới
    last_seq <- c(last_seq[-1], pred)
  }
  
  # Khôi phục dự báo f2 về thang đo ban đầu
  preds_f2 <- preds_scaled * (f2_max - f2_min) + f2_min
  
  #############################
  # 7.4. Dự báo các thành phần khác (f0, f1, d2) và tổng hợp lại kết quả cuối cùng
  #############################
  # Ở đây, ví dụ dùng giá trị cuối quan sát làm dự báo cho f0, f1 và d2
  forecast_f0 <- rep(tail(decomp$f0, 1), forecast_horizon)
  forecast_f1 <- rep(tail(decomp$f1, 1), forecast_horizon)
  forecast_d2 <- rep(tail(decomp$d2, 1), forecast_horizon)
  
  # Kết hợp các dự báo: dự báo cuối cùng cho R = f0 + f1 + f2 (dự báo từ LSTM) + d2
  final_forecast <- forecast_f0 + forecast_f1 + preds_f2 + forecast_d2
  
  #############################
  # 7.5. Vẽ biểu đồ dự báo so với dữ liệu thực tế
  #############################
  actual <- as.numeric(test_data$vnindex_close)
  forecast_dates <- index(test_data)
  
  plot(forecast_dates, actual, type = "l", col = "blue", lwd = 2,
       main = "Dự báo LSTM-LWR vs Thực tế",
       xlab = "Ngày", ylab = "VN-Index",
       ylim = range(c(actual, final_forecast)))
  lines(forecast_dates, final_forecast, col = "red", lwd = 2)
  legend("topleft", legend = c("Thực tế", "LSTM-LWR Dự báo"),
         col = c("blue", "red"), lty = 1, lwd = 2)
  
  # Lưu kết quả dự báo vào danh sách results (nếu bạn đã dùng biến results ở các mô hình khác)
  results[["LSTM_LWR"]] <- final_forecast
  
} else {
  cat("Keras chưa khả dụng. Vui lòng cài đặt và cấu hình keras (với TensorFlow) để chạy mô hình LSTM-LWR.\n")
}


###########################
# 8. Vẽ biểu đồ với tất cả mô hình
###########################
plot(forecast_dates, actual, type="l", col="blue", lwd=2,
     main="Dự báo VN-Index với ARIMA, GARCH và LSTM",
     xlab="Ngày", ylab="VN-Index",
     ylim=range(c(actual, unlist(results))))

lines(forecast_dates, results[["ARIMA"]], col="gray", lwd=2, lty=1)
lines(forecast_dates, results[["GARCH"]], col="black", lwd=2, lty=5)
lines(forecast_dates, results[["ARIMA_GARCH"]], col="darkorange", lwd=2, lty=2)
lines(forecast_dates, results[["ARIMA_EGARCH"]], col="purple", lwd=2, lty=3)
lines(forecast_dates, results[["ARIMA_GJR_GARCH"]], col="green", lwd=2, lty=4)
if ("LSTM" %in% names(results)) {
  lines(forecast_dates, results[["LSTM"]], col="red", lwd=2, lty=6)
  
}

if ("LSTM_DLWR" %in% names(results)) {
  lines(forecast_dates, results[["LSTM_DLWR"]], col="magenta", lwd=2, lty=7)
}

legend("topleft",
       legend=c("Thực tế", "ARIMA", "GARCH", "ARIMA-GARCH", "ARIMA-EGARCH", "ARIMA-GJR-GARCH", "LSTM", "LSTM-DLWR"),
       col=c("blue", "gray", "black", "darkorange", "purple", "green", "red", "magenta"),
       lty=c(1,1,5,2,3,4,6,7), lwd=2)
