import torch
import dlc_practical_prologue as prologue

train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(10)

print(train_input[1])
print(train_target)
print(train_classes)