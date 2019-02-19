import torch

def train(net, batch, batch_size,truth_vector, learning_rate):
    criterion = torch.nn.CrossEntropyLoss()
    #Todo: Switch from SGD
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = net(batch)
    # Compute and print loss
    loss = criterion(y_pred, torch.max(truth_vector, 1)[1])

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def test(net, testing_batch_array, truth_vector):
    batch_size = len(testing_batch_array)
    criterion = torch.nn.CrossEntropyLoss()
    n_correct, n_total = 0, 0
    for i, batch in enumerate(testing_batch_array):
        # Forward pass: Compute predicted y by passing batch to the model
        answer = net(batch[None, :, :, :])
        label = torch.max(truth_vector[i], 0)[1].view(1)
        n_correct += (torch.max(answer, 1)[1].view(1) == label).sum().item()
        n_total += batch_size
    train_acc = 100. * n_correct/batch_size
    print("Training accuracy: ", train_acc)
    return train_acc
   
