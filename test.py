import torch
import np
first_label = torch.tensor([0,1,0])
sec_label = torch.tensor([0,1,1])

ground_truth = torch.stack([first_label ,sec_label], dim=0)

input1 = torch.randn(1, 10, 49)
input2 = torch.randn(1, 10, 49)

mini_batch = torch.stack([input1, input2], dim=0)

# a = []
# for i in range(64):
#     a.append(input1)

# b = torch.Tensor(64, 1, 10, 49)
# torch.cat(a, out=b)



a = []
label = []
for i in range(128):
    a.append(input1.numpy())
    label.append(first_label)
#b = torch.stack(a)
#label_total = torch.stack(label)

#an array of #.wav/64 so 2x(64, 1, 10, 19)
mini_batches = np.array_split(a,len(a)/64)

d