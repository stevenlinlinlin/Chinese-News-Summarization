import numpy as np
import matplotlib.pyplot as plt
import sys
import json

with open(sys.argv[1], 'r') as f:
    data = json.load(f)

log_history = data['log_history']

# rouge curves
x = [epoch['epoch'] for epoch in log_history if 'eval_loss' in epoch.keys()]

plt.figure(0)
plt.title("train/rouge-1")
## rouge-1_f
y = [epoch['eval_rouge-1_f'] for epoch in log_history if 'eval_rouge-1_f' in epoch.keys()]
plt.plot(x, y, label="rouge-1_f")

## rouge-1_p
y = [epoch['eval_rouge-1_p'] for epoch in log_history if 'eval_rouge-1_p' in epoch.keys()]
plt.plot(x, y, label="rouge-1_p")

## rouge-1_r
y = [epoch['eval_rouge-1_r'] for epoch in log_history if 'eval_rouge-1_r' in epoch.keys()]
plt.plot(x, y, label="rouge-1_r")

plt.legend()
plt.savefig('rouge-1_plot.png')

plt.figure(1)
plt.title("train/rouge-2")
## rouge-2_f
y = [epoch['eval_rouge-2_f'] for epoch in log_history if 'eval_rouge-2_f' in epoch.keys()]
plt.plot(x, y, label="rouge-2_f")

## rouge-2_p
y = [epoch['eval_rouge-2_p'] for epoch in log_history if 'eval_rouge-2_p' in epoch.keys()]
plt.plot(x, y, label="rouge-2_p")

## rouge-2_r
y = [epoch['eval_rouge-2_r'] for epoch in log_history if 'eval_rouge-2_r' in epoch.keys()]
plt.plot(x, y, label="rouge-2_r")

plt.legend()
plt.savefig('rouge-2_plot.png')

plt.figure(2)
plt.title("train/rouge-l")
## rouge-l_f
y = [epoch['eval_rouge-l_f'] for epoch in log_history if 'eval_rouge-l_f' in epoch.keys()]
plt.plot(x, y, label="rouge-l_f")

## rouge-l_p
y = [epoch['eval_rouge-l_p'] for epoch in log_history if 'eval_rouge-l_p' in epoch.keys()]
plt.plot(x, y, label="rouge-l_p")

## rouge-l_r
y = [epoch['eval_rouge-l_r'] for epoch in log_history if 'eval_rouge-l_r' in epoch.keys()]
plt.plot(x, y, label="rouge-l_r")

plt.legend()
plt.savefig('rouge-l_plot.png')


