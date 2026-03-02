# 测试所有核心包是否能正常导入
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import sklearn

# 打印版本信息，验证导入成功
print("numpy版本：", np.__version__)
print("pandas版本：", pd.__version__)
print("scipy版本：", scipy.__version__)
print("matplotlib版本：", plt.matplotlib.__version__)
print("sklearn版本：", sklearn.__version__)
print("="*50)
print("所有包导入成功！问题已修复！")