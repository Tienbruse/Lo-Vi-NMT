# File: modules/optim/__init__.py

import torch # Thêm import torch nếu các file optimizer con chưa import

# Giả sử AdamOptim và AdaBeliefOptim là các class đã được định nghĩa trong các file tương ứng
# Hoặc chúng là alias của các optimizer từ torch.optim
from modules.optim.adam import AdamOptim
from modules.optim.adabelief import AdaBeliefOptim

# Import ScheduledOptim từ file scheduler.py trong cùng thư mục modules/optim
# Bỏ comment dòng này để transformer.py có thể import được ScheduledOptim
from modules.optim.scheduler import ScheduledOptim

# Import AdamW trực tiếp từ torch.optim
from torch.optim import AdamW

# Định nghĩa dictionary các optimizer có sẵn
optimizers = {
    "Adam": AdamOptim,          # Giữ lại Adam gốc của bạn
    "AdaBelief": AdaBeliefOptim,  # Giữ lại AdaBelief của bạn
    "AdamW": AdamW              # Thêm AdamW vào dictionary
}

# Lớp ScheduledOptim bây giờ đã được import và có thể được sử dụng bởi các module khác
# khi chúng import từ 'modules.optim'.
