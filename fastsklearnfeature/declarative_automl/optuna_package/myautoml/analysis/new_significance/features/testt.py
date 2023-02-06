import heapq
my_list = [6,7,8,9,10]

def get_val(val):
    return val

best_trials = heapq.nsmallest(2, my_list, key=get_val)

print(best_trials)