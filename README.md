*Nếu cần sử dụng GPU NVIDIA để huấn luyện mô hình  
Cài đặt CUDA Toolkit 11.x/12.x trên hệ thống  
Cài đặt cuDNN Library for Windows (ZIP) từ [https://developer.nvidia.com/rdp/cudnn-archive ](https://developer.nvidia.com/cudnn)   
Cài đặt gói torch và kiểm tra liên kết với CUDA  
install.packages("torch") 
library(torch)  
Cài torch với CUDA 11.8 (khớp với bản nvcc bạn có)  
install_torch(cuda_version = "11.8")
