from sklearn.utils.class_weight import compute_sample_weight

groups = ['a', 'a', 'b', 'b', 'c']

print(compute_sample_weight(class_weight='balanced', y=groups))