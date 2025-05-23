# 1. Cài đặt và nạp các gói cần thiết
packages <- c("readxl", "xts", "zoo", "tseries", "rugarch", "forecast", "FinTS", "Rlibeemd")
installed <- rownames(installed.packages())
for (pkg in packages) {
  if (!(pkg %in% installed)) install.packages(pkg)
  library(pkg, character.only = TRUE)
}

# 2. Đọc và xử lý dữ liệu
data <- read_excel("data_vnindex1.xlsx")
names(data) <- tolower(names(data))
data$date <- as.Date(data$date)
data <- data[order(data$date), ]
data_xts <- xts(data[, setdiff(names(data), "date")], order.by = data$date)

# 3. Chia dữ liệu thành tập huấn luyện và kiểm tra
n <- 90  # số ngày cuối dùng làm test
total_rows <- nrow(data_xts)

training_data <- data_xts[1:(total_rows - n), ]
test_data     <- data_xts[(total_rows - n + 1):total_rows, ]

# Tạo log-return
log_return_train <- diff(log(training_data$vnindex_close))
log_return_train <- na.omit(log_return_train)

log_return_test <- diff(log(test_data$vnindex_close))
log_return_test <- na.omit(log_return_test)

# Giá cuối cùng của tập huấn luyện (điểm khởi đầu tính giá)
P0 <- as.numeric(tail(training_data$vnindex_close, 1))

cat("\n--- Kiểm định tính dừng ---\n")
cat("ADF test:\n")
print(adf.test(log_return_train))
cat("\nKPSS test:\n")
print(kpss.test(log_return_train))

# 4. Ước lượng mô hình ARIMA
fit_arima <- auto.arima(log_return_train)
print(summary(fit_arima))
arima_order <- arimaorder(fit_arima)

# 5. Định nghĩa các mô hình GARCH biến thể
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

# 6. Dự báo log-return cho các mô hình
forecast_horizon <- length(log_return_test)
models <- list(
  ARIMA = fit_arima,
  GARCH = spec_garch,
  ARIMA_GARCH = spec_arima_garch,
  ARIMA_EGARCH = spec_arima_egarch,
  ARIMA_GJR_GARCH = spec_arima_gjr
)

log_return_preds <- list()
price_preds <- list()

# ARIMA: dự báo log-return rồi chuyển sang giá
forecast_arima <- forecast(models$ARIMA, h = forecast_horizon)
log_return_preds[["ARIMA"]] <- as.numeric(forecast_arima$mean)
price_preds[["ARIMA"]] <- exp(cumsum(log_return_preds[["ARIMA"]])) * P0

# GARCH-type models rolling forecast
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
  log_return_preds[[name]] <- mu
  price_preds[[name]] <- exp(cumsum(mu)) * P0
}

# Tính lại giá thực tế từ log-return test
actual_price <- exp(cumsum(as.numeric(log_return_test))) * P0

# Hàm đánh giá
r_squared <- function(y_true, y_pred) {
  ss_res <- sum((y_true - y_pred)^2, na.rm = TRUE)
  ss_tot <- sum((y_true - mean(y_true, na.rm = TRUE))^2, na.rm = TRUE)
  1 - ss_res / ss_tot
}

mape <- function(y_true, y_pred) {
  eps <- 1e-6
  denominator <- ifelse(abs(y_true) < eps, eps, y_true)
  mean(abs((y_true - y_pred) / denominator), na.rm = TRUE) * 100
}

rmse <- function(y_true, y_pred) {
  sqrt(mean((y_true - y_pred)^2, na.rm = TRUE))
}

mae <- function(y_true, y_pred) {
  mean(abs(y_true - y_pred), na.rm = TRUE)
}

# 7. Đánh giá mô hình dựa trên giá
metrics <- data.frame(
  Model = character(),
  R2 = numeric(),
  MAPE = numeric(),
  RMSE = numeric(),
  MAE = numeric(),
  stringsAsFactors = FALSE
)

for (name in names(price_preds)) {
  pred_price <- price_preds[[name]]
  actual_trimmed <- actual_price[1:length(pred_price)]
  
  metrics <- rbind(metrics, data.frame(
    Model = name,
    R2    = r_squared(actual_trimmed, pred_price),
    MAPE  = mape(actual_trimmed, pred_price),
    RMSE  = rmse(actual_trimmed, pred_price),
    MAE   = mae(actual_trimmed, pred_price)
  ))
}

print(metrics)

