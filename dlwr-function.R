###########################
# 1. Cài đặt và nạp các gói cần thiết
###########################
packages <- c("readxl", "xts", "zoo", "tseries", "rugarch", "forecast",
              "Rlibeemd","torch","locfit","purrr")
installed <- rownames(installed.packages())
for (pkg in packages) {
  if (!(pkg %in% installed)) install.packages(pkg)
  library(pkg, character.only = TRUE)
}

###########################
# 2. Đọc và xử lý dữ liệu
###########################
data <- read_excel("data_vnindex1.xlsx")
names(data) <- tolower(names(data))
data$date <- as.Date(data$date)
data <- data[order(data$date), ]
data_xts <- xts(data[, setdiff(names(data), "date")], order.by = data$date)

# Kiểm định tính dừng chuỗi log-return
log_return_train <- diff(log(data$vnindex_close))
log_return_train <- na.omit(log_return_train)

cat("\n--- Kiểm định tính dừng ---\n")
cat("ADF test:\n")
print(adf.test(log_return_train))
cat("\nKPSS test:\n")
print(kpss.test(log_return_train))

# Cài đặt nếu chưa có:
# install.packages("locfit")


# 1) Hàm DLWR cơ bản: dùng n quan sát (window) gần nhất để fit locfit.robust
dlwr_fit <- function(x, y, n, deg = 1, alpha = 0.3, cens=1, kern = "tcub",maxit = 30,start_t = 1) {
  # x, y: vector, x là thời gian/index, y là giá trị
  # n: số quan sát trong "cửa sổ" (window) cho mỗi lần fit
  # deg: bậc đa thức cục bộ
  # alpha: độ trơn/bandwidth (tương đương span) của locfit
  # kern: hàm trọng số, mặc định là tricube theo Cleveland 1979, trang 833
  #maxit: số lần lặp, tham số t trong Cleveland 1979
  
  N <- length(x)
  if (length(y) != N) stop("Chiều dài x và y phải bằng nhau.")
  
  y_fitted <- rep(NA, N)
  
  for (t in seq(start_t, N)) {
  
    
    # Lấy n điểm gần nhất từ t-(n-1) đến t
    idx_start <- t - (n - 1)

    # Nếu idx_start < 1 => chưa đủ window
    if (idx_start < 1) next
    
    idx_end   <- t

    #Thông báo theo dõi tiến trình
    cat("\n--- t =", t, " idx_start=", idx_start, " idx_end=", idx_end, "---\n")

    x_sub <- x[idx_start:idx_end]
    cat("x_sub = ", x_sub, "\n")
    
    y_sub <- y[idx_start:idx_end]
    cat("y_sub = ", y_sub, "\n")
    
    # Nếu t nằm trong khoảng từ 11 đến 30, in ra x_sub và y_sub để kiểm tra
    if(t==39){
      cat("\n--- Debug t =", t, " (idx_start =", idx_start, " idx_end =", idx_end, ") ---\n")
      cat("x_sub:", paste(x_sub, collapse = " "), "\n")
      cat("y_sub:", paste(y_sub, collapse = " "), "\n")
    }
    
    
    # Kiểm tra NA
    if (anyNA(x_sub) || anyNA(y_sub)) {
      cat(">> PHÁT HIỆN NA trong x_sub hoặc y_sub! <<\n")
      cat("x_sub:", x_sub, "\n")
      cat("y_sub:", y_sub, "\n")
      cat("Which x_sub are NA? ", which(is.na(x_sub)), "\n")
      cat("Which y_sub are NA? ", which(is.na(y_sub)), "\n")
      next
    }
    
    #cat("d0[1:30] = ", y_sub[1:30], "\n")  # Bên ngoài, xem 30 phần tử đầu
    #cat("y_sub = ", y_sub[1:30], "\n")
    
    # Khớp mô hình local regression robust
    fit <- locfit.robust(y_sub ~ lp(x_sub, deg = deg), alpha = alpha, weights = rep(1, length(y_sub)), cens=rep(0, length(y_sub)),kern = kern, maxit = maxit)
    
    # Dự đoán riêng cho điểm cuối (thời điểm t) - tránh dùng tương lai
    y_hat_t <- predict(fit, newdata = data.frame(x_sub = x[t]))
    y_fitted[t] <- y_hat_t
  }
  
  return(y_fitted)
}


# 2) Hàm tách 3 lần (3 vòng) 
dlwr_separation_3 <- function(data,
                              date_col,                      # tên cột ngày
                              value_col,                     # tên cột giá trị
                              cens  = 0,
                              n     =20,    # vector 3 giá trị n cho 3 vòng
                              deg   = 1,       # bậc đa thức cho 3 vòng
                              alpha = 0.2, # bandwidth (span) cho 3 vòng
                              kern  ="tcub",
                              maxit = 20){
  # data: data.frame chứa (date_col, value_col)
  # date_col: cột ngày
  # value_col: cột giá trị (chẳng hạn close price)
  # n_vec, deg_vec, alpha_vec: 3 phần tử tương ứng 3 lần tách
  # ...: tham số khác cho locfit.robust (kernel, maxk, v.v.)
  
  # Bước 1: Sắp xếp data theo date_col
  data <- data[order(data[[date_col]]), ]
  
  # Bước 2: Tạo x và R
  x <- seq_len(nrow(data))  # index 1, 2, ..., N (hoặc dùng as.numeric(data[[date_col]]) nếu cần)
  R <- data[[value_col]]
  N <- length(R)
  

  # -- Lần tách thứ nhất --
  f0 <- dlwr_fit(x, R,
                 n     = n,
                 cens  = cens,
                 deg   = deg,
                 alpha = alpha,
                 kern  = kern,
                 maxit = maxit,
                 start_t = 1)
  d0 <- R - f0
  
  # -- Lần tách thứ hai --

  f1 <- dlwr_fit(x, d0,
                 n     = n,
                 cens  = cens,
                 deg   = deg,
                 alpha = alpha,
                 kern  = kern,
                 maxit = maxit,
                 start_t = 2*n-1)
  d1 <- d0 - f1


  # -- Lần tách thứ ba --
  f2 <- dlwr_fit(x, d1,
                 n     = n,
                 cens  = cens,
                 deg   = deg,
                 alpha = alpha,
                 kern  = kern,
                 maxit = maxit,
                 start_t = 3*n-2)
  d2 <- d1 - f2
  
  # Tính giá trị xấp xỉ chuỗi gốc
  R_approx <- f0 + f1 + f2 + d2
  
  # Trả kết quả dưới dạng data frame hoặc list
  result_df <- data.frame(
    date      = data[[date_col]],
    R         = R,
    f0        = f0,
    f1        = f1,
    f2        = f2,
    d2        = d2,
    R_approx  = R_approx
  )
  
  return(result_df)
}

res_sep <- dlwr_separation_3(
  data       = data,
  date_col   = "date",
  value_col  = "vnindex_close",
  n      = 20, # có thể điều chỉnh
  deg    = 1,    # ví dụ: lần 1,2 bậc 2; lần 3 bậc 1
  alpha  = 0.2,
  kern   = "tcub",
  maxit = 30
)

res_sep <- na.omit(res_sep)

# Xem kết quả
print(res_sep[58:70,])

#############


train_lstm_for_dlwr <- function(res_sep,
                                lookback = 10,
                                horizon_weeks = c(1, 2, 3, 4),
                                epochs = 20,
                                batch_size = 16,
                                lr = 0.001,
                                test_ratio = 0.7) {
  
  device <- torch_device(if (cuda_is_available()) "cuda" else "cpu")
  
  # Hàm tạo tập dữ liệu
  create_dataset <- function(series, lookback, device) {
    series <- torch_tensor(series)$to(dtype = torch_float(), device = device)
    X <- list()
    Y <- list()
    for (i in 1:(length(series) - lookback)) {
      x_i <- series[i:(i + lookback - 1)]$unsqueeze(2)
      y_i <- series[i + lookback]
      X[[i]] <- x_i
      Y[[i]] <- y_i
    }
    list(
      x = torch_stack(X),
      y = torch_stack(Y)
    )
  }
  
  # Mạng LSTM theo mô tả bài báo
  Net <- nn_module(
    "Net",
    initialize = function() {
      self$lstm1 <- nn_lstm(1, 64, batch_first = TRUE)
      self$lstm2 <- nn_lstm(64, 128, batch_first = TRUE)
      self$lstm3 <- nn_lstm(128, 128, batch_first = TRUE)
      self$dense1 <- nn_linear(128, 128)
      self$dropout <- nn_dropout(p = 0.2)
      self$dense2 <- nn_linear(128, 1)
    },
    forward = function(x) {
      x <- self$lstm1(x)[[1]]
      x <- self$lstm2(x)[[1]]
      x <- self$lstm3(x)[[1]]
      x <- x[ , dim(x)[2], ]
      x <- self$dense1(x)
      x <- nnf_relu(x)
      x <- self$dropout(x)
      self$dense2(x)
    }
  )
  
  train_component <- function(series, lookback, horizon, epochs, batch_size, lr) {
    N <- length(series)
    n_train <- floor((1 - test_ratio) * N)
    train_series <- series[1:n_train]
    test_series <- series[(n_train - lookback + 1):N]
    
    train_data <- create_dataset(train_series, lookback, device)
    test_data <- create_dataset(test_series, lookback, device)
    
    model <- Net()$to(device = device)
    optimizer <- optim_adam(model$parameters, lr = lr)
    loss_fn <- nn_mse_loss()
    
    train_ds <- tensor_dataset(train_data$x, train_data$y)
    train_dl <- dataloader(train_ds, batch_size = batch_size, shuffle = TRUE)
    
    for (epoch in 1:epochs) {
      model$train()
      coro::loop(for (batch in train_dl) {
        optimizer$zero_grad()
        output <- model(batch[[1]])
        loss <- loss_fn(output, batch[[2]]$unsqueeze(2))
        loss$backward()
        optimizer$step()
      })
    }
    
    model$eval()
    with_no_grad({
      preds <- model(test_data$x)
    })
    
    preds <- as.numeric(preds$squeeze())
    return(preds[(length(preds) - horizon + 1):length(preds)])
  }
  
  # Dự báo từng thành phần
  result_table <- data.frame()
  for (h in horizon_weeks) {
    horizon <- h * 5
    cat(paste0("\n-- Dự báo ", h, " tuần (", horizon, " ngày) --\n"))
    
    pf0 <- train_component(res_sep$f0, lookback, horizon, epochs, batch_size, lr)
    pf1 <- train_component(res_sep$f1, lookback, horizon, epochs, batch_size, lr)
    pf2 <- train_component(res_sep$f2, lookback, horizon, epochs, batch_size, lr)
    pd2 <- train_component(res_sep$d2, lookback, horizon, epochs, batch_size, lr)
    
    r_hat <- pf0 + pf1 + pf2 + pd2
    true <- tail(res_sep$R, horizon)
    
    mae <- mean(abs(true - r_hat), na.rm = TRUE)
    rmse <- sqrt(mean((true - r_hat)^2, na.rm = TRUE))
    mape <- mean(abs((true - r_hat) / true), na.rm = TRUE) * 100
    
    result_table <- rbind(result_table, data.frame(
      Horizon = paste(h, "tuần"),
      MAE = round(mae, 2),
      RMSE = round(rmse, 2),
      MAPE = round(mape, 2)
    ))
  }
  
  return(result_table)
}


#--- 1. Khai báo Hyperparameters ---
dlwr_params <- list(
  n = c(15, 20, 25),       
  deg = c(1, 2),             
  alpha = c(0.2, 0.3, 0.4),  
  kern = "tcub",             
  maxit = 30                 
)

lstm_params <- list(
  lookback = c(5, 10, 15),         
  epochs = c(20, 50),               
  batch_size = c(16, 32),           
  lr = c(0.001, 0.005),             
  horizon_weeks = 1:4,              
  test_ratio = 0.7                  
)

#--- 2. Hàm chạy một scenario ---
run_scenario <- function(dlwr_p, lstm_p, data){
  
  res_sep <- dlwr_separation_3(
    data       = data,
    date_col   = "date",
    value_col  = "vnindex_close",
    n          = dlwr_p$n,
    deg        = dlwr_p$deg,
    alpha      = dlwr_p$alpha,
    kern       = dlwr_p$kern,
    maxit      = dlwr_p$maxit
  )
  
  res_sep <- na.omit(res_sep)
  
  result <- train_lstm_for_dlwr(
    res_sep = res_sep,
    lookback = lstm_p$lookback,
    horizon_weeks = lstm_p$horizon_weeks,
    epochs = lstm_p$epochs,
    batch_size = lstm_p$batch_size,
    lr = lstm_p$lr,
    test_ratio = lstm_p$test_ratio
  )
  
  return(result)
}

#--- 3. Hàm auto-tune ---
auto_tune_model <- function(dlwr_params, lstm_params, data){
  
  param_grid <- expand.grid(
    n = dlwr_params$n,
    deg = dlwr_params$deg,
    alpha = dlwr_params$alpha,
    lookback = lstm_params$lookback,
    epochs = lstm_params$epochs,
    batch_size = lstm_params$batch_size,
    lr = lstm_params$lr
  )
  
  results <- list()
  
  for(i in seq_len(nrow(param_grid))){
    cat("\n\n--- Scenario:", i, "/", nrow(param_grid), "---\n")
    scenario <- param_grid[i, ]
    
    dlwr_p <- list(
      n = scenario$n,
      deg = scenario$deg,
      alpha = scenario$alpha,
      kern = dlwr_params$kern,
      maxit = dlwr_params$maxit
    )
    
    lstm_p <- list(
      lookback = scenario$lookback,
      epochs = scenario$epochs,
      batch_size = scenario$batch_size,
      lr = scenario$lr,
      horizon_weeks = lstm_params$horizon_weeks,
      test_ratio = lstm_params$test_ratio
    )
    
    result <- run_scenario(dlwr_p, lstm_p, data)
    
    results[[i]] <- list(
      scenario = scenario,
      performance = result
    )
    
    print(result)
  }
  
  return(results)
}

#--- Chạy hàm tự động fine-tune ---
final_results <- auto_tune_model(dlwr_params, lstm_params, data)
