#--- 1. Load thư viện ---
libs <- c("readxl",  "torch", "locfit", "purrr", "doParallel", "foreach")

# Kiểm tra và cài đặt nếu thiếu
missing_libs <- setdiff(libs, installed.packages()[,"Package"])
if(length(missing_libs)) install.packages(missing_libs)

# Sau đó nạp thư viện
invisible(lapply(libs, library, character.only = TRUE))

#-----------------------------------------------------------------------------------------

#--- 2. Hàm tiền xử lý dữ liệu ---
prepare_data <- function(filepath){
  data <- read_excel(filepath)
  names(data) <- tolower(names(data))
  data$date <- as.Date(data$date)
  data <- data[order(data$date), ]
  return(data)
}

#-----------------------------------------------------------------------------------------

#--- 3. Các hàm chức năng

#Hàm DLWR_Fit: tính hồi quy cục bộ động có trọng số cho chuỗi thời gian
dlwr_fit <- function(x, y, n, deg = 1, alpha = 0.3, kern = "gauss", maxit = 30, start_t = 1){
  N <- length(x)
  y_fitted <- rep(NA, N)
  
  idx_start <- pmax(1, seq(start_t, N) - (n - 1))
  idx_end <- seq(start_t, N)
  
  fits <- map2(idx_start, idx_end, function(s, e){
    if(e - s + 1 < n) return(NA)
    x_sub <- x[s:e]
    y_sub <- y[s:e]
    
    if(anyNA(x_sub) || anyNA(y_sub) || any(is.infinite(x_sub)) || any(is.infinite(y_sub))){
      return(NA)
    }
    
    fit <- tryCatch({
      locfit.robust(y_sub ~ lp(x_sub, deg), alpha=alpha, kern=kern, maxit=maxit)
    }, error = function(e) NULL)
    
    # Kiểm tra xem fit có thành công không
    if(is.null(fit) || !inherits(fit, "locfit")) return(NA)
    
    # Dự báo điểm cuối cùng
    pred <- predict(fit, newdata = data.frame(x_sub = x[e]))
    if(length(pred) != 1 || any(is.na(pred))) return(NA)
    
    pred
  })
  
  y_fitted[idx_end[!is.na(fits)]] <- unlist(fits[!is.na(fits)])
  return(y_fitted)
}


#Hàm DLWR_separation: phân rã chuỗi thời gian thành các thành phần nhằm dự báo
dlwr_separation <- function(data, date_col, value_col, params){
  data <- data[order(data[[date_col]]), ]
  x <- seq_len(nrow(data))
  R <- data[[value_col]]
  
  # Vòng 1: start_t = 1
  f0 <- dlwr_fit(x, R, params$n, params$deg, params$alpha, start_t=1)
  d0 <- R - f0
  
  # Vòng 2: start_t = 2*n - 1
  f1 <- dlwr_fit(x, d0, params$n, params$deg, params$alpha, start_t=2*params$n - 1)
  d1 <- d0 - f1
  
  # Vòng 3: start_t = 3*n - 2
  f2 <- dlwr_fit(x, d1, params$n, params$deg, params$alpha, start_t=3*params$n - 2)
  d2 <- d1 - f2
  
  cut_idx <- 3 * params$n - 3
  
  result_df <- data.frame(date=data[[date_col]], R, f0, f1, f2, d2)
  result_df <- result_df[-seq_len(cut_idx), ]
  return(na.omit(result_df))
}

# Chuẩn hóa MinMax về [0,1] để dùng cho deep learning
minmax_scale <- function(x) {
  min_val <- min(x, na.rm = TRUE)
  max_val <- max(x, na.rm = TRUE)
  scaled <- (x - min_val) / (max_val - min_val)
  list(scaled = scaled, min = min_val, max = max_val)
}

# Khôi phục lại giá trị gốc sau khi dự báo bằng deep learning
minmax_inverse <- function(scaled_x, min_val, max_val) {
  scaled_x * (max_val - min_val) + min_val
}  

#tạo bộ dữ liệu phù hợp cho deep learning
create_dataset <- function(series, lookback){
  
  series_tensor <- torch_tensor(series, dtype = torch_float())
  series_tensor <- series_tensor$to(device = device)
  X <- lapply(1:(length(series_tensor)-lookback), function(i) series_tensor[i:(i+lookback-1)]$unsqueeze(2))
  Y <- series_tensor[(lookback+1):length(series_tensor)]
  list(x = torch_stack(X), y = Y)
}

#Chia thành tập train/test/validation
split_series <- function(series, train_ratio = 0.6, val_ratio = 0.1) {
  N <- length(series)
  idx_train <- floor(N * train_ratio)
  idx_val   <- floor(N * val_ratio)
  
  if ((idx_train + idx_val) >= N) {
    stop("Tập dữ liệu quá ngắn để tách thành train/val/test theo tỷ lệ đã cho.")
  }
  
  list(
    train = series[1:idx_train],
    val   = series[(idx_train + 1):(idx_train + idx_val)],
    test  = series[(idx_train + idx_val + 1):N]
  )
}


#Tính chỉ tiêu đo lường hiệu quả mô hình: R^2
r_squared <- function(y_true, y_pred) {
  ss_res <- sum((y_true - y_pred)^2, na.rm = TRUE)
  ss_tot <- sum((y_true - mean(y_true, na.rm = TRUE))^2, na.rm = TRUE)
  1 - ss_res / ss_tot
}


# Định nghĩa các thành phần cho học sâu: LSTM
train_lstm_component <- function(series, lookback, epochs, batch_size, lr, device){
  
  print(paste("Check inputs: lookback =", lstm_p$lookback,
              "epochs =", lstm_p$epochs,
              "batch_size =", lstm_p$batch_size,
              "lr =", lstm_p$lr))


  
  # Kiểm tra độ dài chuỗi phải lớn hơn lookback
  if(length(series) <= lookback){
    warning("Chuỗi không đủ dữ liệu sau khi làm sạch.")
    return(rep(NA, length(series)))
  }
  
  # Bước 2: MinMax normalization
  scaled <- minmax_scale(series)
  scaled_series <- scaled$scaled
  min_val <- scaled$min
  max_val <- scaled$max
  
  # Tách tập train/val/test
  splits <- split_series(scaled_series, 0.6, 0.1)
  train <- splits$train
  val <- splits$val
  test <- splits$test
  
  ds_train <- create_dataset(train,lookback)
  ds_val <- create_dataset(val,lookback)
  ds_test <- create_dataset(test,lookback)
  
  
  # Dataloader
  train_dl <- dataloader(tensor_dataset(ds_train$x, ds_train$y), batch_size=batch_size, shuffle=TRUE)
  val_dl <- dataloader(tensor_dataset(ds_val$x, ds_val$y), batch_size=batch_size)
  
  
  model <- LSTMModel()$to(device=device)
  optimizer <- optim_adam(model$parameters, lr = lr)
  loss_fn <- nn_mse_loss()
  
  model$train()
  
 for(epoch in seq_len(epochs)){
    coro::loop(for(b in train_dl){
      optimizer$zero_grad()
      loss <- loss_fn(model(b[[1]])$squeeze(), b[[2]])
      loss$backward()
      optimizer$step()
    })
  }
  
  model$eval()
  val_losses <- c()
  
  with_no_grad({
    coro::loop(for (b in val_dl) {
      output <- model(b[[1]])$squeeze()
      loss <- loss_fn(output, b[[2]])
      val_losses <- c(val_losses, as.numeric(loss))
    })
  })
  
  avg_val_loss <- mean(val_losses)
  cat(sprintf("Epoch %d/%d | Validation Loss: %.4f\n", epoch, epochs, avg_val_loss))
 

  model$eval()
  with_no_grad({
  preds <- model(ds_test$x)$cpu()$squeeze()
  })
    
  preds <- as.numeric(preds)
  summary(preds)
  return(list(preds = preds, min = min_val, max = max_val))
}

#Thiết kế module LSTM phù hợp
LSTMModel <- nn_module(
  "LSTMModel",
  initialize = function() {
    self$lstm1 <- nn_lstm(input_size = 1, hidden_size = 64, batch_first = TRUE)
    self$lstm2 <- nn_lstm(input_size = 64, hidden_size = 128, batch_first = TRUE)
    self$lstm3 <- nn_lstm(input_size = 128, hidden_size = 128, batch_first = TRUE)
    self$fc1 <- nn_linear(128, 128)
    self$dropout <- nn_dropout(0.2)
    self$fc2 <- nn_linear(128, 1)
  },
  forward = function(x) {
    x <- self$lstm1(x)[[1]]
    x <- self$lstm2(x)[[1]]
    x <- self$lstm3(x)[[1]]
    x <- x[ , dim(x)[2], ] # Lấy timestep cuối cùng
    x <- self$fc1(x)
    x <- nnf_relu(x)
    x <- self$dropout(x)
    x <- self$fc2(x)
    x <- nnf_relu(x)
  }
)


#--- 6. Auto-tune Parallel ---
auto_tune_parallel <- function(dlwr_params, lstm_params, raw_data){
  
  param_grid <- expand.grid(
  n = dlwr_params$n,
  deg = dlwr_params$deg,
  alpha = dlwr_params$alpha,
  kern = dlwr_params$kern,
  lookback = lstm_params$lookback,
  epochs = lstm_params$epochs,
  batch_size = lstm_params$batch_size,
  lr = lstm_params$lr
)

  
  cl <- makeCluster(detectCores() - 1)
  registerDoParallel(cl)
  
  results <- foreach(
    i = seq_len(nrow(param_grid)), 
    .packages = c('torch', 'locfit', 'purrr'),
    .export = c("dlwr_separation", "train_lstm_component", "dlwr_fit", "LSTMModel",
                "minmax_scale", "minmax_inverse", "r_squared","create_dataset","split_series")
  ) %dopar% {
    dlwr_p <- param_grid[i, c('n','deg','alpha',"kern")]
    lstm_p <- param_grid[i, c('lookback','epochs','batch_size','lr')]
    

    res_sep <- dlwr_separation(raw_data, "date", "vnindex_close", dlwr_p)
    
    device <- torch_device(if (cuda_is_available()) "cuda" else "cpu")

    pf0_raw <- train_lstm_component(res_sep$f0, lstm_p$lookback, lstm_p$epochs, lstm_p$batch_size, lstm_p$lr, device)
    pf1_raw <- train_lstm_component(res_sep$f1, lstm_p$lookback, lstm_p$epochs, lstm_p$batch_size, lstm_p$lr, device)
    pf2_raw <- train_lstm_component(res_sep$f2, lstm_p$lookback, lstm_p$epochs, lstm_p$batch_size, lstm_p$lr, device)
    pd2_raw <- train_lstm_component(res_sep$d2, lstm_p$lookback, lstm_p$epochs, lstm_p$batch_size, lstm_p$lr, device)
    
    # Inverse từng phần
    pf0 <- minmax_inverse(pf0_raw$preds, pf0_raw$min, pf0_raw$max)
    pf1 <- minmax_inverse(pf1_raw$preds, pf1_raw$min, pf1_raw$max)
    pf2 <- minmax_inverse(pf2_raw$preds, pf2_raw$min, pf2_raw$max)
    pd2 <- minmax_inverse(pd2_raw$preds, pd2_raw$min, pd2_raw$max)
    
    # Cộng thành tổng dự báo
    r_hat <- pf0 + pf1 + pf2 + pd2

    true <- tail(res_sep$R, length(r_hat))
    
    list(
      params = param_grid[i, ],
      MAE  = mean(abs(true - r_hat), na.rm = TRUE),
      RMSE = sqrt(mean((true - r_hat)^2, na.rm = TRUE)),
      MAPE = mean(abs((true - r_hat) / true), na.rm = TRUE) * 100,
      R2   = r_squared(true, r_hat) 
    )
  }
  
  
  stopCluster(cl)
  saveRDS(results, "grid_search_results.rds")
  return(results)
}

#--- 7. Chạy pipeline hoàn chỉnh ---
raw_data <- prepare_data("data_vnindex1.xlsx")
dlwr_params <- list(n=c(15,20), deg=c(1,2), alpha=0.2, kern = c("tricube", "gauss"))
lstm_params <- list(lookback=c(5,10,15,20), epochs=50, batch_size=16, lr=c(0.0005,0.001))

results <- auto_tune_parallel(dlwr_params, lstm_params, raw_data)
print(results)



###############
param_grid <- data.frame(
  n = 15,
  deg = 2,
  kern = "tricub",
  alpha = 0.2,
  lookback = 10,
  epochs = 20,
  batch_size = 16,
  lr = 0.001
)

# Lấy tham số ra
dlwr_p <- list(
  n     = param_grid$n[1],
  deg   = param_grid$deg[1],
  kern  = param_grid$kern[1],
  alpha = param_grid$alpha[1]
)
lstm_p <- list(
  lookback    = param_grid$lookback[1],
  epochs      = param_grid$epochs[1],
  batch_size  = param_grid$batch_size[1],
  lr          = param_grid$lr[1]
)

# Tạo tập DLWR
raw_data <- prepare_data("data_vnindex1.xlsx")
res_sep <- dlwr_separation(raw_data, "date", "vnindex_close", dlwr_p)
cat("Range f0:", range(res_sep$f0, na.rm = TRUE), "\n")
cat("Range f1:", range(res_sep$f1, na.rm = TRUE), "\n")
cat("Range f2:", range(res_sep$f2, na.rm = TRUE), "\n")
cat("Range d2:", range(res_sep$d2, na.rm = TRUE), "\n")
device <- torch_device(if (cuda_is_available()) "cuda" else "cpu")
summary(res_sep)
# Dự báo từng thành phần (trả về list chứa preds, min, max)
summary(res_sep$f1)
summary(res_sep$d2)
pf0_raw <- train_lstm_component(res_sep$f0, lstm_p$lookback, lstm_p$epochs, lstm_p$batch_size, lstm_p$lr, device)
pf1_raw <- train_lstm_component(res_sep$f1, lstm_p$lookback, lstm_p$epochs, lstm_p$batch_size, lstm_p$lr, device)
pf2_raw <- train_lstm_component(res_sep$f2, lstm_p$lookback, lstm_p$epochs, lstm_p$batch_size, lstm_p$lr, device)
pd2_raw <- train_lstm_component(res_sep$d2, lstm_p$lookback, lstm_p$epochs, lstm_p$batch_size, lstm_p$lr, device)
 

# Inverse từng phần
pf0 <- minmax_inverse(pf0_raw$preds, pf0_raw$min, pf0_raw$max)
pf1 <- minmax_inverse(pf1_raw$preds, pf1_raw$min, pf1_raw$max)
pf2 <- minmax_inverse(pf2_raw$preds, pf2_raw$min, pf2_raw$max)
pd2 <- minmax_inverse(pd2_raw$preds, pd2_raw$min, pd2_raw$max)

# Tổng hợp dự báo
r_hat <- pf0 + pf1 + pf2 + pd2
true <- tail(res_sep$R, length(r_hat))

# Đánh giá
result <- list(
  params = param_grid[1, ],
  MAE  = mean(abs(true - r_hat), na.rm = TRUE),
  RMSE = sqrt(mean((true - r_hat)^2, na.rm = TRUE)),
  MAPE = mean(abs((true - r_hat) / true), na.rm = TRUE) * 100,
  R2   = r_squared(true, r_hat)
)

print(result)



