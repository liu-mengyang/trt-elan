import matplotlib.pyplot as plt
import numpy as np


print('Plot figure 6: Accuracy results.')

items = ["pytorch",
         "trt fp32",
         "trt tf32",
         "trt fp16",
         "trt ptq",
         "trt qat"]

x = np.arange(len(items))

pytorch = 359
trt_fp32 = 301
trt_tf32 = 259
trt_fp16 = 168
trt_int8_ptq = 142
trt_int8_qat = 142

width = 0.8
n = 1

font_size = 14
plt.rc('font',**{'size': font_size, 'family': 'Arial' })
# plt.rc('pdf',fonttype = 42)

fig = plt.figure()
ax = fig.add_subplot(111)

rects1 = ax.bar(x[0], pytorch, width/n, label="Pytorch",
                edgecolor='black', linewidth=1, color='k')
rects2 = ax.bar(x[1], trt_fp32, width/n, label="TensorRT fp32",
                edgecolor='black', linewidth=1, color='tab:blue')
rects3 = ax.bar(x[2], trt_tf32, width/n, label="TensorRT tf32",
                edgecolor='black', linewidth=1, color='tab:orange')
rects4 = ax.bar(x[3], trt_fp16, width/n, label="TensorRT fp16",
                edgecolor='black', linewidth=1, color='tab:green')
rects5 = ax.bar(x[4], trt_int8_ptq, width/n, label="TensorRT int8 ptq",
                edgecolor='black', linewidth=1, color='tab:red')
rects6 = ax.bar(x[5], trt_int8_qat, width/n, label="TensorRT int8 qat",
                edgecolor='black', linewidth=1, color='tab:purple')

ax.set_ylim(0, 35)

# ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ax.tick_params(bottom=False)
d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
# ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

ax.set_ylabel('Latency (ms)')
ax.set_xticks(x)
ax.set_xticklabels(items)
# ax.legend(frameon=False,ncol=2,loc='upper left', bbox_to_anchor=(0.0, 0.8, 0.5, 0.5),
#           prop={'size': font_size-1})
ax.legend(ncol=2, loc='upper left')
plt.savefig("fig3.png", format="png")