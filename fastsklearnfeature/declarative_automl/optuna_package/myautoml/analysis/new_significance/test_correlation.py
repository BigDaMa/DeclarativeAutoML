from scipy.stats import mannwhitneyu


pop_static = [0.5] *10
pop_dyn = [0.6] *10

w, p = mannwhitneyu(pop_static, pop_dyn, alternative='less')
print('w: ' + str(w) + ' p: ' + str(p))