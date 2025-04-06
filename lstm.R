###########################
# 1. Cài đặt và nạp các gói cần thiết
###########################
packages <- c("readxl", "xts", "zoo", "tseries", "rugarch", "forecast", "FinTS","Rlibeemd","torch")
installed <- rownames(installed.packages())
for (pkg in packages) {
  if (!(pkg %in% installed)) install.packages(pkg)
  library(pkg, character.only = TRUE)
}
tf <- tensorflow::tf
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
device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")
cat("Thiết bị đang dùng:", device$type, "\n")

# ------------------------------
# 1. Dữ liệu & CEEMDAN phân rã
# ------------------------------
raw_series <- as.numeric(training_data$vnindex_close)
forecast_horizon <- as.integer(nrow(test_data))
imfs <- ceemdan(raw_series, ensemble_size = 250L, noise_strength = 0.2)
num_imfs <- min(nrow(imfs), 6)  # Giới hạn số IMFs để tăng hiệu năng

# ------------------------------
# 2. Hàm chuẩn hóa MinMax
# ------------------------------
scale_minmax <- function(x) {
  min_x <- min(x, na.rm = TRUE)
  max_x <- max(x, na.rm = TRUE)
  range <- max_x - min_x
  if (range == 0) range <- 1
  scaled <- (x - min_x) / range
  list(scaled = scaled, min = min_x, max = max_x)
}

# ------------------------------
# 3. Chuẩn bị dữ liệu LSTM
# ------------------------------
create_lstm_data <- function(series, lag = 10L) {
  X <- list()
  y <- c()
  for (i in 1:(length(series) - lag)) {
    X[[i]] <- series[i:(i + lag - 1)]
    y[i] <- series[i + lag]
  }
  X_array <- array(unlist(X), dim = c(length(X), lag, 1))
  y_array <- y
  X_tensor <- torch_tensor(X_array, dtype = torch_float())$to(device = device)
  y_tensor <- torch_tensor(y_array, dtype = torch_float())$to(device = device)
  list(X = X_tensor, y = y_tensor)
}

# ------------------------------
# 4. Mô hình LSTM đơn giản
# ------------------------------
lstm_model <- nn_module(
  initialize = function() {
    self$lstm <- nn_lstm(input_size = 1, hidden_size = 32, batch_first = TRUE)
    self$fc <- nn_linear(32, 1)
  },
  forward = function(x) {
    out <- self$lstm(x)[[1]]
    last <- out[ , dim(out)[2], ]
    self$fc(last)
  }
)

# ------------------------------
# 5. Huấn luyện và dự báo 1 IMF
# ------------------------------
train_predict_lstm <- function(
    scaled_series,
    lag = 10L,
    horizon = 30L,
    epochs = 100L,
    batch_size = 16L,
    early_stopping_patience = 10,
    min_delta = 1e-5,
    verbose = TRUE
) {
  # 1. Tạo dữ liệu huấn luyện
  data <- create_lstm_data(scaled_series, lag)
  X <- data$X
  y <- data$y
  
  # 2. Khởi tạo mô hình và tối ưu
  model <- lstm_model()
  model$to(device = device)
  optimizer <- optim_adam(model$parameters, lr = 0.001)
  loss_fn <- nn_mse_loss()
  
  best_loss <- Inf
  patience_counter <- 0
  
  # 3. Huấn luyện mô hình
  for (epoch in 1:epochs) {
    model$train()
    optimizer$zero_grad()
    
    output <- model(X)
    output <- output$view(c(-1))  # chuyển về vector cùng shape với y
    
    loss <- loss_fn(output, y)
    loss$backward()
    
    nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0)
    optimizer$step()
    
    current_loss <- loss$item()
    if (verbose && epoch %% 10 == 0) {
      cat(sprintf("Epoch %d: loss = %.6f\n", epoch, current_loss))
    }
    
    if (best_loss - current_loss > min_delta) {
      best_loss <- current_loss
      patience_counter <- 0
    } else {
      patience_counter <- patience_counter + 1
    }
    
    if (patience_counter >= early_stopping_patience) {
      if (verbose) cat("Early stopping tại epoch", epoch, "\n")
      break
    }
  }
  
  # 4. Dự báo chuỗi tương lai
  model$eval()
  preds <- numeric(horizon)
  last_seq <- tail(scaled_series, lag)
  
  for (i in 1:horizon) {
    last_seq <- as.numeric(last_seq)
    
    if (length(last_seq) != lag) {
      stop("Lỗi: Độ dài chuỗi không đúng tại bước ", i)
    }
    
    if (any(!is.finite(last_seq))) {
      stop("Lỗi: Chuỗi chứa giá trị không hợp lệ tại bước ", i)
    }
    
    input_array <- array(last_seq, dim = c(1, lag, 1))
    mode(input_array) <- "numeric"
    
    if (typeof(input_array) != "double") {
      stop("Lỗi: input_array không phải kiểu double tại bước ", i)
    }
    
    input_tensor <- tryCatch({
      torch_tensor(input_array, dtype = torch_float())$to(device = device)
    }, error = function(e) {
      stop("Lỗi khi tạo tensor tại bước ", i, ": ", conditionMessage(e))
    })
    
    pred <- as.numeric(model(input_tensor)$item())
    
    if (!is.finite(pred)) {
      stop("Lỗi: Dự báo không hợp lệ tại bước ", i)
    }
    
    preds[i] <- pred
    last_seq <- c(last_seq[-1], pred)
  }
  
  # 5. Giải phóng bộ nhớ GPU nếu cần
  rm(model)
  gc()
  torch::cuda_empty_cache()
  
  return(preds)
}



# ------------------------------
# 6. Huấn luyện từng IMF và tổng hợp
# ------------------------------
preds_components <- list()
for (i in 1:num_imfs) {
  cat("📦 Dự báo IMF", i, "\n")
  comp <- imfs[i, ]
  scaled <- scale_minmax(comp)
  pred_scaled <- train_predict_lstm(
    scaled_series = scaled$scaled,
    lag = 10L,
    horizon = forecast_horizon,
    epochs = 50
  )
  
  # Phục hồi thang đo ban đầu
  pred_unscaled <- pred_scaled * (scaled$max - scaled$min) + scaled$min
  preds_components[[paste0("IMF", i)]] <- pred_unscaled
}

# ------------------------------
# 7. Tổng hợp dự báo cuối cùng
# ------------------------------
final_forecast <- Reduce("+", preds_components)
results[["CEEMDAN_LSTM"]] <- final_forecast



library(keras)
library(tensorflow)

#########################
# 1. Hàm phân tách DLWR
#########################
DLWR_decompose <- function(R, spans = c(0.3, 0.3, 0.3), degree = 2) {
  n <- length(R)
  t <- seq_len(n)
  fit_f0 <- loess(R ~ t, span = spans[1], degree = degree)
  f0 <- predict(fit_f0, newdata = data.frame(t = t))
  d0 <- R - f0
  fit_f1 <- loess(d0 ~ t, span = spans[2], degree = degree)
  f1 <- predict(fit_f1, newdata = data.frame(t = t))
  d1 <- d0 - f1
  fit_f2 <- loess(d1 ~ t, span = spans[3], degree = degree)
  f2 <- predict(fit_f2, newdata = data.frame(t = t))
  d2 <- d1 - f2
  return(list(f0 = f0, f1 = f1, f2 = f2, d2 = d2))
}

#########################
# 2. Tiện ích hỗ trợ
#########################
scale_minmax <- function(x) {
  min_x <- min(x, na.rm = TRUE)
  max_x <- max(x, na.rm = TRUE)
  scaled <- (x - min_x) / (max_x - min_x)
  list(scaled = scaled, min = min_x, max = max_x)
}

create_lstm_data <- function(series, lag = 10L) {
  n <- length(series)
  num_samples <- n - lag
  X <- array(NA, dim = c(num_samples, lag, 1))
  y <- array(NA, dim = c(num_samples))
  for (i in 1:num_samples) {
    X[i, , 1] <- series[i:(i + lag - 1)]
    y[i] <- series[i + lag]
  }
  list(X = X, y = y)
}

build_lstm_model <- function(input_shape = c(10L, 1L)) {
  model <- keras_model_sequential()
  model$add(layer_lstm(units = 64, input_shape = input_shape, return_sequences = TRUE))
  model$add(layer_lstm(units = 128, return_sequences = TRUE))
  model$add(layer_lstm(units = 128))
  model$add(layer_dense(units = 128, activation = "relu"))
  model$add(layer_dropout(rate = 0.2))
  model$add(layer_dense(units = 1, activation = "relu"))
  return(model)
}



train_predict_lstm <- function(scaled_series, lag = 10L, horizon = 30L, epochs = 100, batch_size = 32) {
  data <- create_lstm_data(scaled_series, lag)
  X <- data$X
  y <- data$y
  
  model <- build_lstm_model(input_shape = c(lag, 1))
  model$compile(
    loss = "mean_squared_error",
    optimizer = keras$optimizers$Adam()
  )
  
  early_stop <- callback_early_stopping(
    monitor = "loss",
    patience = 10,
    restore_best_weights = TRUE
  )
  
  model$fit(
    x = X,
    y = y,
    epochs = 100L,
    batch_size = 32L,
    callbacks = list(early_stop)
  )
  
  preds <- numeric(horizon)
  last_seq <- tail(scaled_series, lag)
  for (i in 1:horizon) {
    input_seq <- array(last_seq, dim = c(1L, lag, 1L))
    input_seq <- tf$cast(input_seq, dtype = tf$float32)
    pred <- as.numeric(model$predict(input_seq))
    preds[i] <- pred
    last_seq <- c(last_seq[-1], pred)
  }
  
  return(preds)
}

#########################
# 3. Bắt đầu triển khai mô hình DLWR-LSTM
#########################

# Dữ liệu đầu vào
raw_series <- as.numeric(training_data$vnindex_close)
decomp <- DLWR_decompose(raw_series, spans = c(0.3, 0.3, 0.3), degree = 2)

# Số bước dự báo
forecast_horizon <- as.integer(nrow(test_data))

# Khởi tạo danh sách kết quả
preds_components <- list()

# Huấn luyện và dự báo từng thành phần DLWR bằng LSTM riêng
for (name in c("f0", "f1", "f2", "d2")) {
  comp <- decomp[[name]]
  scaled <- scale_minmax(comp)
  pred_scaled <- train_predict_lstm(
    scaled_series = scaled$scaled,
    lag = 10L,
    horizon = forecast_horizon,
    epochs = 100,
    batch_size = 32
  )
  preds_components[[name]] <- pred_scaled * (scaled$max - scaled$min) + scaled$min
}

# Tổng hợp dự báo cuối cùng
final_forecast <- preds_components$f0 + preds_components$f1 + preds_components$f2 + preds_components$d2

#########################
# 4. Vẽ kết quả so với thực tế
#########################

actual <- as.numeric(test_data$vnindex_close)
forecast_dates <- index(test_data)

plot(forecast_dates, actual, type = "l", col = "blue", lwd = 2,
     main = "Dự báo DLWR-LSTM với 4 thành phần",
     xlab = "Ngày", ylab = "VN-Index",
     ylim = range(c(actual, final_forecast)))
lines(forecast_dates, final_forecast, col = "red", lwd = 2)
legend("topleft", legend = c("Thực tế", "DLWR-LSTM Dự báo"),
       col = c("blue", "red"), lty = 1, lwd = 2)

# Lưu kết quả
results[["LSTM_LWR"]] <- final_forecast



library(torch)
library(Rlibeemd)

# 1. Dữ liệu và thiết lập
raw_series <- as.numeric(training_data$vnindex_close)
forecast_horizon <- as.integer(nrow(test_data))

# 2. CEEMDAN phân rã chuỗi
imfs <- ceemdan(raw_series, ensemble_size = 250L, noise_strength = 0.2)
num_imfs <- nrow(imfs)
max_imfs <- min(num_imfs, 6)  # chỉ dùng 6 IMF đầu
preds_components <- list()

# 3. Scale MinMax
scale_minmax <- function(x) {
  min_x <- min(x, na.rm = TRUE)
  max_x <- max(x, na.rm = TRUE)
  scaled <- (x - min_x) / (max_x - min_x)
  list(scaled = scaled, min = min_x, max = max_x)
}

# 4. Tạo dữ liệu cho LSTM
create_lstm_data <- function(series, lag = 10L) {
  n <- length(series)
  X <- list()
  y <- c()
  for (i in 1:(n - lag)) {
    X[[i]] <- series[i:(i + lag - 1)]
    y[i] <- series[i + lag]
  }
  X_tensor <- torch_tensor(array(unlist(X), dim = c(length(X), lag, 1)), dtype = torch_float())
  y_tensor <- torch_tensor(y, dtype = torch_float())
  list(X = X_tensor, y = y_tensor)
}

# 5. Mô hình LSTM
lstm_model <- nn_module(
  initialize = function() {
    self$lstm1 <- nn_lstm(input_size = 1, hidden_size = 64, batch_first = TRUE)
    self$lstm2 <- nn_lstm(input_size = 64, hidden_size = 128, batch_first = TRUE)
    self$fc1 <- nn_linear(128, 64)
    self$drop <- nn_dropout(p = 0.2)
    self$fc2 <- nn_linear(64, 1)
  },
  forward = function(x) {
    x <- self$lstm1(x)[[1]]
    x <- self$lstm2(x)[[1]]
    x <- x[ , dim(x)[2], ]
    x <- nnf_relu(self$fc1(x))
    x <- self$drop(x)
    self$fc2(x)
  }
)

# 6. Huấn luyện LSTM trên từng IMF
train_predict_lstm_torch <- function(scaled_series, lag = 10L, horizon = 30L,
                                     epochs = 30, batch_size = 16, device = torch_device("cpu")) {
  data <- create_lstm_data(scaled_series, lag)
  X <- data$X$to(device = device)
  y <- data$y$to(device = device)
  
  model <- lstm_model()
  model$to(device = device)
  
  optimizer <- optim_adam(model$parameters, lr = 0.01)
  loss_fn <- nn_mse_loss()
  
  for (epoch in 1:epochs) {
    model$train()
    optimizer$zero_grad()
    pred <- model(X)$squeeze()
    loss <- loss_fn(pred, y)
    loss$backward()
    optimizer$step()
    if (epoch %% 10 == 0) cat(sprintf("Epoch %d - Loss: %.6f\n", epoch, loss$item()))
  }
  
  # Dự báo
  preds <- c()
  last_seq <- tail(scaled_series, lag)
  model$eval()
  for (i in 1:horizon) {
    input <- torch_tensor(array(last_seq, dim = c(1, lag, 1)), dtype = torch_float())$to(device = device)
    pred <- model(input)
    pred_val <- as.numeric(pred$item())
    preds[i] <- pred_val
    last_seq <- c(last_seq[-1], pred_val)
  }
  
  rm(model); gc(); torch::cuda_empty_cache()
  return(preds)
}

# 7. Dự báo lần lượt 6 IMF đầu
device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")
for (i in 1:max_imfs) {
  cat(sprintf("🧠 Huấn luyện IMF %d / %d\n", i, max_imfs))
  comp <- imfs[i, ]
  scaled <- scale_minmax(comp)
  pred_scaled <- train_predict_lstm_torch(
    scaled_series = scaled$scaled,
    lag = 10L,
    horizon = forecast_horizon,
    epochs = 30,
    batch_size = 16,
    device = device
  )
  preds_components[[i]] <- pred_scaled * (scaled$max - scaled$min) + scaled$min
}

# 8. Tổng hợp dự báo
final_forecast <- Reduce(`+`, preds_components)

# 9. Vẽ kết quả
actual <- as.numeric(test_data$vnindex_close)
forecast_dates <- index(test_data)

plot(forecast_dates, actual, type = "l", col = "blue", lwd = 2,
     main = "CEEMDAN-LSTM torch (6 IMF)",
     xlab = "Ngày", ylab = "VN-Index",
     ylim = range(c(actual, final_forecast)))
lines(forecast_dates, final_forecast, col = "red", lwd = 2)
legend("topleft", legend = c("Thực tế", "CEEMDAN-LSTM (6 IMF)"),
       col = c("blue", "red"), lwd = 2)

# 10. Lưu kết quả
results[["CEEMDAN_LSTM_6IMF"]] <- final_forecast






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

if ("LSTM_DLWR" %in% names(results)) {
  lines(forecast_dates, results[["LSTM_DLWR"]], col="magenta", lwd=2, lty=7)
}

legend("topleft",
       legend=c("Thực tế", "ARIMA", "GARCH", "ARIMA-GARCH", "ARIMA-EGARCH", "ARIMA-GJR-GARCH", "LSTM", "LSTM-DLWR"),
       col=c("blue", "gray", "black", "darkorange", "purple", "green", "red", "magenta"),
       lty=c(1,1,5,2,3,4,6,7), lwd=2)
