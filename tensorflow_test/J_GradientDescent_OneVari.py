# -*- coding: cp936 -*-
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# ����ѵ������
x = np.arange(0., 10., 0.2)
m = len(x)                                      # ѵ�����ݵ���Ŀ
print(m)
x0 = np.full(m, 1.0)
input_data = np.vstack([x0, x]) .T              # ��ƫ��b��ΪȨ�����ĵ�һ������
target_data = 2 * x + 5 + np.random.randn(m)
# ������ֹ����
loop_max = 10000   # ����������(��ֹ��ѭ��)
epsilon = 1e-3
# ��ʼ��Ȩֵ
np.random.seed(0)
w = np.random.randn(2)
#w = np.zeros(2)
alpha = 0.001      # ����(ע��ȡֵ����ᵼ����,��С�����ٶȱ���)
diff = 0.
error = np.zeros(2)
count = 0          # ѭ������
finish = 0         # ��ֹ��־
# -------------------------------------------����ݶ��½��㷨----------------------------------------------------------
while count < loop_max:
    count += 1
    # ����ѵ�����ݼ������ϸ���Ȩֵ
    for i in range(m):
        diff = np.dot(w, input_data[i]) - target_data[i]  # ѵ��������,�������ֵ

        # ��������ݶ��½��㷨,����һ��Ȩֵֻʹ��һ��ѵ������
        w = w - alpha * diff * input_data[i]
        # ------------------------------��ֹ�����ж�-----------------------------------------
        # ��û��ֹ���������ȡ�������д������������������ȡ�����,��ѭ�����´�ͷ��ʼ��ȡ�������д���

    # ----------------------------------��ֹ�����ж�-----------------------------------------
    # ע�⣺�ж��ֵ�����ֹ���������ж�����λ�á���ֹ�жϿ��Է���Ȩֵ��������һ�κ�,Ҳ���Է��ڸ���m�κ�
    if np.linalg.norm(w - error) < epsilon:     # ��ֹ������ǰ�����μ������Ȩ�����ľ��������С
        finish = 1
        break
    else:
        error = w
print('loop count = %d' % count+ '\tw:[%f, %f]' % (w[0], w[1]))
# -----------------------------------------------�ݶ��½���-----------------------------------------------------------
'''
while count < loop_max:
    count += 1
    # ��׼�ݶ��½�����Ȩֵ����ǰ����������������������ݶ��½���Ȩֵ��ͨ������ĳ��ѵ�����������µ�
    # �ڱ�׼�ݶ��½��У�Ȩֵ���µ�ÿһ���Զ��������ͣ���Ҫ����ļ���
    sum_m = np.zeros(2)
    for i in range(m):
        dif = (np.dot(w, input_data[i]) - target_data[i]) * input_data[i]
        sum_m = sum_m + dif  # ��alphaȡֵ����ʱ,sum_m���ڵ��������л����
    w = w - alpha * sum_m  # ע�ⲽ��alpha��ȡֵ,����ᵼ����
    # �ж��Ƿ�������
    if np.linalg.norm(w - error) < epsilon:
        finish = 1
        break
    else:
        error = w
print('loop count = %d' % count+ '\tw:[%f, %f]' % (w[0], w[1]))
'''
# check with scipy linear regression
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x, target_data)
print('intercept = %s slope = %s' % (intercept, slope))
plt.plot(x, target_data, 'k+')
plt.plot(x, w[1] * x + w[0], 'r')
plt.show()
