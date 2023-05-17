# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from functools import partial
# from copy import deepcopy
# torch.manual_seed(0)

# # Here's a simple MLP
# class SimpleMLP(nn.Module):
#     def __init__(self):
#         super(SimpleMLP, self).__init__()
#         self.fc1 = nn.Linear(784, 128)
#         self.batch_norm1 = nn.BatchNorm1d(128)
#         self.fc2 = nn.Linear(128, 128)
#         self.batch_norm2 = nn.BatchNorm1d(128)
#         self.fc3 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = x.flatten(1)
        
#         x = self.fc1(x)
#         x = self.batch_norm1(x)
#         x = F.relu(x)
        
#         x = self.fc2(x)
#         x = self.batch_norm2(x)
#         x = F.relu(x)
        
#         x = self.fc3(x)
#         return x
    

# ''''''
# device = 'cuda'
# num_models = 11
# data = torch.randn(100, 6, 1, 28, 28, device=device)
# targets = torch.randint(10, (6400,), device=device)
# models = [SimpleMLP().to(device) for _ in range(num_models)]

# model_num = 3
# single_model = deepcopy(models[model_num])

# ''''''
# minibatches = data[:num_models]
# predictions_diff_minibatch_loop = [model(minibatch) for model, minibatch in zip(models, minibatches)]

# # minibatch = data[0]
# # predictions2 = [model(minibatch) for model in models]

# ''''''
# from functorch import combine_state_for_ensemble
# fmodel, params, buffers = combine_state_for_ensemble(models)
# [p.requires_grad_() for p in params]

# # # show the leading 'num_models' dimension
# # print([p.size(0) for p in params]) 
# # # verify minibatch has leading dimension of size 'num_models'
# # assert minibatches.shape == (num_models, 64, 1, 28, 28) 

# ''' different model, different minibatch '''
# from functorch import vmap
# predictions1_vmap = vmap(fmodel)(params, buffers, minibatches)

# print(f'predictions1_vmap shape: {predictions1_vmap.shape}')
# assert torch.allclose(predictions1_vmap, torch.stack(predictions_diff_minibatch_loop), atol=1e-3, rtol=1e-5)


# predictions1_vmap_first = predictions1_vmap.reshape(6*num_models,10)
# index = torch.tensor( [i*6 for i in range(num_models)] )
# predictions1_vmap_first = predictions1_vmap_first[index]
# print(f'predictions1_vmap_first shape: {predictions1_vmap_first.shape}')

# loss = torch.norm(predictions1_vmap_first, dim=1).sum()
# loss.backward()



# the_data = minibatches[model_num]
# output = single_model(the_data)
# print(f'output shape: {output.shape}')
# output = output[0,:]
# tmp_loss = torch.norm(output)
# tmp_loss.backward()


# for p_vmap, p_single in zip(params, single_model.parameters()):
#     print(f'p_vmap shape: {p_vmap.shape}')
#     print(f'p_single shape: {p_single.shape}')
#     print(f'p_vmap.grad shape: {p_vmap.grad.shape}')
#     print(f'p_single.grad shape: {p_single.grad.shape}')
#     print(f'p_vmap.grad: {p_vmap.grad}')
#     print(f'p_single.grad: {p_single.grad}')
#     assert torch.allclose(p_vmap.grad[model_num], p_single.grad, atol=1e-3, rtol=1e-5)

# print('Done with no error')





# import torch
# from functorch import grad
# def my_loss_func(y, y_pred, something):
#    loss_per_sample = (0.5 * y_pred - y) ** 2
# #    the_sum = 0
# #    for p in something:
# #        the_sum += p
#    loss = loss_per_sample.mean() + something[0]
#    return loss, (y_pred, loss_per_sample)

# fn = grad(my_loss_func, argnums=(0,1,2), has_aux=True)
# y_true = torch.rand(4)
# print(y_true.requires_grad)
# y_preds = torch.rand(4, requires_grad=True)
# out = fn(y_true, y_preds, [torch.tensor(1.0, requires_grad=True), torch.tensor(2.0, requires_grad=True)])
# [print(index, i) for index, i in enumerate(out)]






# ''' different model, same minibatch '''
# predictions2_vmap = vmap(fmodel, in_dims=(0, 0, None))(params, buffers, minibatch)
# assert torch.allclose(predictions2_vmap, torch.stack(predictions2), atol=1e-3, rtol=1e-5)

# from torch.utils.benchmark import Timer
# without_vmap = Timer(
#     stmt="[model(minibatch) for model, minibatch in zip(models, minibatches)]",
#     globals=globals())
# with_vmap = Timer(
#     stmt="vmap(fmodel)(params, buffers, minibatches)",
#     globals=globals())
# print(f'Predictions without vmap {without_vmap.timeit(100)}')
# print(f'Predictions with vmap {with_vmap.timeit(100)}')


from datasets import imdb
imdb.main()


# def iterator():
#     for i in range(10):
#         yield i

#     for j in range(10):
#         yield j

# for i in iterator():
#     print(i)