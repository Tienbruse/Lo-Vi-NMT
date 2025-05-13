# File: modules/optim/__init__.py

import torch # Thêm import torch nếu các file optimizer con chưa import

# Giả sử AdamOptim và AdaBeliefOptim là các class đã được định nghĩa trong các file tương ứng
# Hoặc chúng là alias của các optimizer từ torch.optim
from modules.optim.adam import AdamOptim
from modules.optim.adabelief import AdaBeliefOptim
# scheduler.py được import ở nơi khác (ví dụ: transformer.py) nên không cần thiết ở đây
# from modules.optim.scheduler import ScheduledOptim

# Import AdamW trực tiếp từ torch.optim
from torch.optim import AdamW

# Định nghĩa dictionary các optimizer có sẵn
optimizers = {
    "Adam": AdamOptim,          # Giữ lại Adam gốc của bạn
    "AdaBelief": AdaBeliefOptim,  # Giữ lại AdaBelief của bạn
    "AdamW": AdamW              # Thêm AdamW vào dictionary
}

# Bạn có thể xóa dòng import ScheduledOptim nếu nó không được dùng trực tiếp trong file __init__ này.
# Lớp ScheduledOptim được sử dụng trong transformer.py để bọc quanh các optimizer trên.
