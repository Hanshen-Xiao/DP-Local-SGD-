                    
import enum
import torch
import time
from tqdm import tqdm
import random
import utility
import torchvision
import torchvision.transforms as T
from functorch import combine_state_for_ensemble, make_functional, make_functional_with_buffers
from functorch import vmap, grad
from copy import deepcopy
import os
import numpy as np
import math
''' '''
import logger

''' helpers '''
TRAIN_SETUP_LIST = ('epoch', 'device', 'optimizer', 'loss_metric', 'enable_per_grad')

mean_list = []
quantile_75_list = []
std_list =[]
quantile_25_list = []


class Phase(enum.Enum):
    TRAIN = enum.auto()
    VAL = enum.auto()
    TEST = enum.auto()
    PHASE_to_PHASE_str = { TRAIN: "Training", VAL: "Validation", TEST: "Testing"}

class train_master:
    def __init__(self, *,
                model,
                loaders = (None, None, None),
                train_setups = dict(),
                # expected_batchsize = None,
                # batch_para_computer_batch_size = None,
                arg_setup = None,
                ):
        self.data_logger = utility.log_master(root = arg_setup.log_dir)
        logger.init_log(dir = arg_setup.log_dir)
        self.arg_setup = arg_setup

        self.data_recorder = logger.data_recorder(f'clip_c{self.arg_setup.C}.json')

        self.model = model   
        self.num_of_classes = self.model.num_of_classes
        self.num_of_models = arg_setup.num_groups
        
        # self.worker_model_func, self.worker_param_func, self.worker_buffers_func = make_functional_with_buffers(deepcopy(self.model), disable_autograd_tracking=True)
        self.num_of_groups = arg_setup.num_groups // self.arg_setup.group_size
        models = [ deepcopy(self.model) for _ in range( self.num_of_groups ) ]
        self.worker_model_func, self.worker_param_func, self.worker_buffers_func = combine_state_for_ensemble(models)
        
        # print(111, len(self.worker_param_func))
        for p in self.worker_param_func:
            p.requires_grad = False
            # print('p shape:', p.shape)
        
        self.num_groups = arg_setup.num_groups
        self.times_larger = self.arg_setup.samples_per_group 

        # print(f'==> duplicating model {self.num_groups} times...', end = '')
        # self.worker_models, self.worker_params, self.worker_buffers = combine_state_for_ensemble( [deepcopy(self.model) for _ in range(self.num_groups)] ) 
        # print('done')

        self.loaders = {'train': loaders[0], 'val': loaders[1], 'test': loaders[2], 'batch_computer_loader': loaders[3]}
        # assert len(self.loaders['train']) == len(self.loaders['batch_computer_loader']), (len(self.loaders['train']), len(self.loaders['batch_computer_loader']))

        self.train_setups = train_setups
        
        self.loss_metric = self.train_setups['loss_metric']
        
        ''' sanity check '''
        if self.loaders['train'] is None and self.loaders['val'] is None and self.loaders['test'] is None:
            raise ValueError('at least one loader must be provided')
        for setup in TRAIN_SETUP_LIST:
            if setup not in self.train_setups:
                raise ValueError(f'{setup} must be provided in train_setups')
        for setup in self.train_setups:
            if setup is None:
                raise ValueError(f'invalid setups (no NONE setup allowed): {self.train_setups}')
        
        ''' processing the model '''
        self.sigma = self.train_setups['sigma']
        logger.write_log(f'==>  sigma: {self.sigma}')
        
        ''' set the optimizer after extension '''
        self.optimizer = self.train_setups['optimizer']
        
        ''''''
        self.count_parameters() 
        
        
        print(f'==> have {torch.cuda.device_count()} cuda devices')
        # print(f'current device: {self.model.device}')
        
        # print('==> initializing the momemtum history container...')
        # self.computing_device = self.model.device
        # self.container_device = torch.device("cuda:1")
        # self.per_momentum_history = torch.zeros(50000, self.total_params, device = self.container_device)

        self.shape_interval = []
        self.shape_list = []
        last = 0
        for p in self.model.parameters():
            if p.requires_grad:
                self.shape_list.append(p.shape)
                total_param_sub = p.numel()
                self.shape_interval.append([last, last + total_param_sub])
                last += total_param_sub
            else:
                self.shape_interval.append(None)
        self.all_indexes = list(range(self.arg_setup.usable_train_data_samples))
        
        
        
        ''' reindexing '''
        ''' [pri, pri, pub, pub, pub, pub] -> [pri, pub, pub, pri, pub, pub] '''
        # self.reindexing = []
        # for i in range(self.num_of_models):
        #     self.reindexing.append(i)
        #     # self.reindexing += list( range(self.num_of_models + i * times_larger, self.num_of_models + (i + 1) * times_larger) )
        #     ''' fetch the data sample from each data batch, in the same position i'''
        #     self.reindexing += [i + self.num_of_models * j for j in range(1, self.times_larger+1)]
        # self.reindexing = torch.tensor(self.reindexing, device = self.model.device)
        
        self.reindexing = self.get_reindex(self.num_of_models)
        
        ''' transformation list '''
        self.transforms = [
                            T.RandomHorizontalFlip(p= 1),
                            # T.RandomCrop(32, padding=4),
                            # *[T.RandomPerspective(distortion_scale = 0.1 + i * (0.7 - 0.1) / self.times_larger, p = 1) 
                            #   for i in range(self.times_larger - 2)],
                            
                            # *[
                            #     T.Compose([
                            #                 #T.RandomCrop(32, padding = 8),
                            #                 T.RandomHorizontalFlip(p=0.5),
                            #                 ]) 
                            #     for i in range(self.arg_setup.samples_per_group )
                            #   ]
                            ]
        
        # assert len(self.transforms) >= self.arg_setup.samples_per_group 
        # self.transforms = self.transforms[:self.arg_setup.samples_per_group ]
        # print(f'==> {len(self.transforms)} transforms are used')
        
        self.grad_momentum = [ torch.zeros_like(p.data) if p.requires_grad else None for p in self.model.parameters()  ]
        self.iterator_check = [0 for _ in self.model.parameters()]
        self.per_grad_momemtum = [ 0 for _ in self.model.parameters()  ]
    
        # self.un_flattened_grad = []
        # ratio = (self.arg_setup.C_insig**0.5 * self.arg_setup.C_ + sig_C**0.5 * all_big_norm) / self.arg_setup.C

        self.norm_choices = [1+0.25*i for i in range(16)]
        self.avg_pnorms_holder = {norm_choice: [] for norm_choice in self.norm_choices}
        self.avg_inverse_pnorms_holder = {norm_choice: [] for norm_choice in self.norm_choices}
        
        
        # '''get whole batch data '''
        # print('==> stacking all train  data...')
        # loader = self.loaders['train']
        # self.whole_data_container = None
        # self.whole_label_container = None
        # self.whole_index_container = None
        # for index, train_batch in enumerate(loader):
        #                 # print(index, end='/')
        #     # if isinstance(train_batch[1], list) and len(train_batch[1]) ==2:
        #     #     data_index = train_batch[1][1]
        #     #     train_batch = (train_batch[0], train_batch[1][0])
        #     #     # batch_para_batch = None
                
        #     ''' get training data '''
        #     inputs, targets = map(lambda x: x.to(self.train_setups['device']), train_batch)
        #     if self.whole_data_container is None:
        #         self.whole_data_container = inputs
        #         self.whole_label_container = targets
        #         # self.whole_index_container = data_index
        #     else:   
        #         self.whole_data_container = torch.cat([self.whole_data_container, inputs], dim=0)
        #         self.whole_label_container = torch.cat([self.whole_label_container, targets], dim=0)
        #         # self.whole_index_container = torch.cat([self.whole_index_container, data_index], dim=0)
        
        loader = self.loaders['test']
        '''get whole test data '''
        print('==> stacking test data...')
        self.whole_data_container_test = None
        self.whole_label_container_test = None
        self.whole_index_container_test = None
        for index, train_batch in enumerate(loader):
                        # print(index, end='/')
            # if isinstance(train_batch[1], list) and len(train_batch[1]) ==2:
            #     data_index = train_batch[1][1]
            #     train_batch = (train_batch[0], train_batch[1][0])
            #     # batch_para_batch = None
                
            ''' get training data '''
            inputs, targets = map(lambda x: x.to(self.train_setups['device']), train_batch)
            if self.whole_data_container_test is None:
                self.whole_data_container_test = inputs
                self.whole_label_container_test = targets
            else:   
                self.whole_data_container_test = torch.cat([self.whole_data_container_test, inputs], dim=0)
                self.whole_label_container_test = torch.cat([self.whole_label_container_test, targets], dim=0)
        print(f'==> test data size: {self.whole_data_container_test.size()}')
        print(f'==> all labels:', set(self.whole_label_container_test.tolist()))
        self.transformation = T.Compose([
                                    my_RandomHorizontalFlip(p = 0.5),
                                    # my_randcrop(32, padding = 4),
                                    ])


        ''' using pub data '''
        self.pub_num = self.arg_setup.pub_num
        self.dummy_index_pub = torch.tensor( [ i for i in range(self.pub_num) ] )
        
        if self.pub_num == 0:
            self.dummy_index_pub = torch.tensor( [] )
        else:
            self.dummy_index_pub_tmp = []
            for _ in range(self.num_groups):
                tmp_index = torch.randint(self.pub_num, (self.times_larger,))
                self.dummy_index_pub_tmp.append( tmp_index)
            self.dummy_index_pub = torch.cat(self.dummy_index_pub_tmp)
        
        print('==> pub data generation done, pub data shape:', self.dummy_index_pub.shape)
        
        '''logging'''
        self.data_logger.write_log(f'weighted_recall.csv', self.arg_setup)
        logger.write_log(f'arg_setup: {self.arg_setup}')
        # for i in range(torch.cuda.device_count()):
        #     logger.write_log(f'\ncuda memory summary for device {i}:\n{torch.cuda.memory_summary(device=f"cuda:{i}", abbreviated=True)}', verbose=True)
    
    def get_reindex(self, real_batch_num):
        ''' reindexing '''
        ''' [pri, pri, pub, pub, pub, pub] -> [pri, pub, pub, pri, pub, pub] '''
        if self.times_larger == 0:
            return torch.tensor( np.arange(real_batch_num) )

        reindexing = []
        for i in range(real_batch_num):
            reindexing.append(i)
            # self.reindexing += list( range(self.num_of_models + i * times_larger, self.num_of_models + (i + 1) * times_larger) )
            ''' fetch the data sample from each data batch, in the same position i'''
            reindexing += [i + self.num_of_models * j for j in range(1, self.times_larger+1)]
        reindexing = torch.tensor(self.reindexing, device = self.model.device)
        return reindexing

    def count_parameters(self):
        total = 0
        cnn_total = 0
        linear_total = 0

        tensor_dic = {}
        for submodule in self.model.modules():
            for s in submodule.parameters():
                if s.requires_grad:
                    if id(s) not in tensor_dic:
                        tensor_dic[id(s)] = 0
                    if isinstance(submodule, torch.nn.Linear):
                            tensor_dic[id(s)] = 1

        for p in self.model.parameters():
            if p.requires_grad:
                total += int(p.numel())
                if tensor_dic[id(p)] == 0:
                    cnn_total += int(p.numel())
                if tensor_dic[id(p)] == 1:
                    linear_total += int(p.numel())

        self.cnn_total = cnn_total
        logger.write_log(f'==>  model parameter summary:')
        logger.write_log(f'     non_linear layer parameter: {self.cnn_total}' )
        self.linear_total = linear_total
        logger.write_log(f'     Linear layer parameter: {self.linear_total}' )
        self.total_params = self.arg_setup.total_para = total
        logger.write_log(f'     Total parameter: {self.total_params}\n' )
        
    # def count_parameters(model):
    #     total = 0
    #     cnn_total = 0
    #     linear_total = 0

    #     tensor_dic = {}
    #     for submodule in model.modules():
    #         for s in submodule.parameters():
    #             if s.requires_grad:
    #                 if id(s) not in tensor_dic:
    #                     tensor_dic[id(s)] = 0
    #                 if isinstance(submodule, torch.nn.Linear):
    #                         tensor_dic[id(s)] = 1

    #     for p in model.parameters():
    #         if p.requires_grad:
    #             total += int(p.numel())
    #             if tensor_dic[id(p)] == 0:
    #                 cnn_total += int(p.numel())
    #             if tensor_dic[id(p)] == 1:
    #                 linear_total += int(p.numel())

    #     print(f'==>  model parameter summary:')
    #     print(f'     non_linear layer parameter: {cnn_total}' )
    #     print(f'     Linear layer parameter: {linear_total}' )
    #     print(f'     Total parameter: {total}\n' )

    def train(self):
        
        s = time.time()
        for epoch in range(self.train_setups['epoch']):
            logger.write_log(f'\n\nEpoch: [{epoch}] '.ljust(11) + '#' * 35)
            ''' lr rate scheduler '''
            self.epoch = epoch
            
            train_metrics, val_metrics, test_metrics = None, None, None
            self.record_data_type = 'weighted_recall'

            
            ''' training '''
            if self.loaders['train'] is not None:
                train_metrics = self.one_epoch(train_or_val = Phase.TRAIN, loader = self.loaders['train'])
                for i in range(torch.cuda.device_count()):
                    logger.write_log(f'\ncuda memory summary for device {i}:\n{torch.cuda.memory_summary(device=f"cuda:{i}", abbreviated=True)}', verbose=False)
            
            ''' validation '''
            if self.loaders['val'] is not None:
                val_metrics = self.one_epoch(train_or_val = Phase.VAL, loader = self.loaders['val'])

            ''' testing '''
            if self.loaders['test'] is not None:
                test_metrics = self.one_epoch(train_or_val = Phase.TEST, loader = self.loaders['test'])

            '''logging data '''
            data_str = (' '*3).join([
                                f'{epoch}',
                                f'{ float( train_metrics.__getattr__(self.record_data_type) ) * 100:.2f}%'.rjust(7)
                                if train_metrics else 'NAN',

                                f'{ float( val_metrics.__getattr__(self.record_data_type) ) * 100:.2f}%'.rjust(7)
                                if val_metrics else 'NAN',

                                f'{ float( test_metrics.__getattr__(self.record_data_type) ) * 100:.2f}%'.rjust(7)
                                if test_metrics else 'NAN',
                                ])
            
            self.data_logger.write_log(f'{self.record_data_type}.csv', data_str)

        # utility.grad_norm_summary(  self.optimizer.param_groups[0]['lr'],
        #                             self.train_setups['epoch'], 
        #                             self.avg_pnorms_holder, 
        #                             self.avg_inverse_pnorms_holder)
        

        ''' ending '''

        self.data_recorder.save()
        logger.write_log(f'\n\n=> TIME for ALL : {time.time()-s:.2f}  secs')
    
    def _per_sample_augmentation(self):
        ''' per sample augmentation '''
        # if self.times_larger == 0:
        #     return torch.tensor([], device=self.model.device), torch.tensor([], dtype=torch.int64, device=self.model.device)
        if self.pub_num == 0:
            return torch.tensor([], device=self.model.device), torch.tensor([], dtype=torch.int64, device=self.model.device)
        tmp_index = np.random.permutation(self.dummy_index_pub)
        # pub_input = self.whole_data_container[-self.pub_num:]
        # pub_target = self.whole_label_container[-self.pub_num:]
        
        # pub_input = self.whole_data_container[tmp_index]
        # pub_target = self.whole_label_container[tmp_index]

        pub_input = self.whole_data_container_test[tmp_index]
        pub_target = self.whole_label_container_test[tmp_index]
        
        # print('pub_input shape:', pub_input.shape,  'pub_target shape:', pub_target.shape)
        return pub_input, pub_target
    
        # all_indexes = np.random.permutation(self.all_indexes)
        # dm_index = all_indexes[self.num_groups * self.arg_setup.samples_per_group:]
        # the_inputs = self.whole_data_container[ dm_index ]
        # the_targets = self.whole_label_container[ dm_index ]
        
        # return the_inputs, the_targets
    
    def sampling_noise_summary(self, index, per_grad):
        grad_flatten = self.flatten_to_rows(per_grad[0].shape[0], per_grad)

        grad_flatten_mean = torch.mean(grad_flatten, dim=0, keepdim=True)
        center_around_mean = grad_flatten - grad_flatten_mean
       
        grad_norm = torch.norm(grad_flatten, dim=1)
        mean_of_grad_norm = grad_norm.mean()
        # grad_norm = grad_norm - float(mean_of_grad_norm)
        print(float(mean_of_grad_norm))
        
        
        
        grad_norm[grad_norm < 0] = 0
        
        sorted_grad_norm = torch.sort(grad_norm)[0]
        
        
        quantile_0_25_50_75_100 = [sorted_grad_norm[0]] + [sorted_grad_norm[int(len(sorted_grad_norm) * i / 4) - 1] for i in range(1,5)]
        quantile_0_25_50_75_100 = [round(float(q),3) for q in quantile_0_25_50_75_100]
        last_time_norm_of_grad_used_to_update_model = torch.cat([p.reshape(1, -1) for p in self.grad_momentum], dim=1).norm()
        sampling_noise = torch.norm(center_around_mean, dim=1).mean()
        per_grad_mean_norm = grad_flatten_mean.norm()

        self.data_recorder.add_record('sampling_noise', float(sampling_noise))
        self.data_recorder.add_record('quantile_0', quantile_0_25_50_75_100[0])
        self.data_recorder.add_record('quantile_25', quantile_0_25_50_75_100[1])
        self.data_recorder.add_record('quantile_50', quantile_0_25_50_75_100[2])
        self.data_recorder.add_record('quantile_75', quantile_0_25_50_75_100[3])
        self.data_recorder.add_record('quantile_100', quantile_0_25_50_75_100[4])
        self.data_recorder.add_record('per_grad_mean_norm', float(per_grad_mean_norm) )
        self.data_recorder.add_record('last_time_norm_of_grad', float(last_time_norm_of_grad_used_to_update_model))

        mean_of_grad_norm = grad_norm.mean()
        logger.write_log(f'    ----> last time norm of grad used to update model: {last_time_norm_of_grad_used_to_update_model:.2f}')
        logger.write_log(f'    ----> sampling noise: {sampling_noise:.2f}')
        logger.write_log(f'    ----> quantile 0, 25, 50, 75, 100: {quantile_0_25_50_75_100}')
        logger.write_log(f'    ----> grad norm mean: {mean_of_grad_norm:.2f}, std: {grad_norm.std():.2f}')
        logger.write_log(f'    ----> norm of avg of per grad: {per_grad_mean_norm:.2f}')
        logger.write_log('\n\n')
        
        mean_list.append(float(mean_of_grad_norm))
        quantile_25_list.append(quantile_0_25_50_75_100[1])
        item_std = float(grad_norm.std())
        std_list.append(item_std)
        quantile_75_list.append(quantile_0_25_50_75_100[3])
        
        # print('mean', mean_list)
        # print('std', std_list)
        # print('quantile_25', quantile_25_list)
        # print('quantile_75', quantile_75_list)

    def get_per_grad(self, inputs, targets):
        ''''''
        # def compute_loss(model_para, buffers,  inputs, targets):
        #     # print(f'inputs shape: {inputs.shape}')
        #     predictions = self.worker_model_func(model_para, buffers, inputs)
        #     # print(f'predictions shape: {predictions.shape}, targets shape: {targets.shape}')
        #     ''' only compute the loss of the first(private) sample '''
        #     predictions = predictions[:1]
        #     targets = targets[:1]
            
        #     loss = self.loss_metric(predictions, targets.flatten()) #* inputs.shape[0]
        #     return loss
        # per_grad = list( vmap(grad(compute_loss), in_dims=(None, None, 0, 0))(self.worker_param_func, self.worker_buffers_func, inputs, targets) )


        ''''''
        def compute_loss(model_para, buffers,  inputs, targets):
            # print(f'inputs shape: {inputs.shape}')
            predictions = self.worker_model_func(model_para, buffers, inputs)
            # print(f'predictions shape: {predictions.shape}, targets shape: {targets.shape}')
            ''' only compute the loss of the first(private) sample '''
            # predictions = predictions[:1]
            # targets = targets[:1]
            
            loss = self.loss_metric(predictions, targets.flatten()) #* inputs.shape[0]
            return loss
        
        def self_aug_per_grad(model_para, buffers, inputs, targets):
            
            init_model = [p.clone() for p in model_para]
            # running_model = [torch.clone(p) for p in model_para]
            
            momemtum = [0 for _ in range(len(model_para))]
            chain_len = self.arg_setup.chain_len
            beta = self.arg_setup.forward_beta
            # if self.epoch < 25: 
            #     lr_0 = 0.01
            # else:
            #     lr_0 = 0.02
            # lr_0 = 0.025
            lr_0 = 0.025
            # lr = self.arg_setup.forward_lr
            for _ in range(chain_len):
                per_grad = grad(compute_loss)(model_para, buffers, inputs, targets)
                # for p in per_grad:
                #     print(p.shape)
                momemtum = [beta*m + g for m, g in zip(momemtum, per_grad)]
                for p_worker, p_momemtum in zip(model_para, momemtum):
                    # print(p_worker.data.shape, p_momemtum.shape)
                    p_worker.add_(- lr_0 * p_momemtum )
                    
            per_grad = [i - p for p, i in zip(model_para, init_model)]
                
            # for _ in range(self.arg_setup.self_aug_times):
            #     t_inputs = self.transformation(inputs)
            #     cur_grad = grad(compute_loss)(model_para, buffers, t_inputs, targets)
            #     per_grad = [p + g for p, g in zip(per_grad, cur_grad)]
            # per_grad = [p / (self.arg_setup.self_aug_times + 1) for p in per_grad]
            
            return list(per_grad)
        
        # print('inputs shape: ', inputs.shape, 'targets shape: ', targets.shape)
        per_grad = vmap( self_aug_per_grad, in_dims=(0, 0, 0, 0) )(self.worker_param_func, self.worker_buffers_func, inputs, targets)
        

        # ''''''
        # def norm_of_per(the_grad):
        #     return torch.norm( torch.cat([p.reshape(1, -1) for p in the_grad], dim=1) ) 
        # def compute_loss(model_para, buffers,  inputs, targets):
        #     predictions = self.worker_model_func(model_para, buffers, inputs)
        #     ''' only compute the loss of the first(private) sample '''
        #     predictions = predictions[:1]
        #     targets = targets[:1]
        #     loss = self.loss_metric(predictions, targets.flatten()) #* inputs.shape[0]
        #     return loss
        # def self_aug_per_grad(model_para, buffers, inputs, targets):
        #     per_grad = grad(compute_loss)(model_para, buffers, inputs, targets)
        #     per_grad = [p.unsqueeze(0) for p in per_grad]
        #     norm_list = norm_of_per(per_grad).unsqueeze(0)
        #     for _ in range(self.arg_setup.self_aug_times):
        #         t_inputs = self.transformation(inputs)
        #         cur_grad = grad(compute_loss)(model_para, buffers, t_inputs, targets)
        #         cur_grad = [p.unsqueeze(0) for p in cur_grad]
        #         norm_list = torch.cat([norm_list, norm_of_per(cur_grad).unsqueeze(0)], dim=0)
        #         per_grad = [torch.cat([p, g], dim = 0) for p, g in zip(per_grad, cur_grad)]
        #     coef = torch.softmax(norm_list, dim=0)
        #     per_grad = [torch.sum(p * coef.reshape(p.shape[0], *[1 for _ in p.shape[1:]]), dim=0) for p in per_grad]
        #     return per_grad
        # per_grad = vmap(self_aug_per_grad, in_dims=(None, None, 0, 0), randomness='same')(self.worker_param_func, self.worker_buffers_func, inputs, targets)
        
        return per_grad
       
        
    def one_epoch(self, *, train_or_val, loader):
        metrics = utility.ClassificationMetrics(num_classes = self.num_of_classes)
        metrics.num_images = metrics.loss = 0 
        is_training = train_or_val is Phase.TRAIN

        
        # print(f'whole data container shape: {self.whole_data_container.shape}')
        # print(f'whole label container shape: {self.whole_label_container.shape}')
        # print(f'whole index container shape: {self.whole_index_container.shape}')
        # print(np.random.permutation(self.dummy_index ))
            
        with torch.set_grad_enabled(is_training):
            self.model.train(is_training)
            s = time.time()
            if is_training: 
                print(f'==> have {len(loader)} iterations in this epoch')
                for index, train_batch in enumerate(loader):
                    ''' get training data '''
                    the_inputs, the_targets = map(lambda x: x.to(self.train_setups['device']), train_batch)
                    # print(f'the_inputs shape: {the_inputs.shape}')
                    # print(f'the_targets shape: {the_targets.shape}')
                # for _ in range( int(self.whole_data_container.shape[0]) // self.num_groups ):
                #     self.optimizer.zero_grad()
                #     # num_sampls = np.random.binomial(self.whole_data_container.shape[0], self.num_groups/self.whole_data_container.shape[0])
                #     dummy_index = np.random.permutation(self.dummy_index)[:self.num_groups]

                #     tmp = (self.whole_data_container[dummy_index]).to(dtype = torch.uint8)
                #     the_inputs = (dms.T_system(tmp)).to(dtype = torch.float32)
                #     the_inputs  = dms.transformation(the_inputs)
                #     # the_inputs =  self.whole_data_container[ dummy_index ] 
                #     the_targets = self.whole_label_container[ dummy_index ]

                    
                    pub_inputs, pub_targets = self._per_sample_augmentation()
                    
                    new_inputs = torch.concat([the_inputs, pub_inputs], dim = 0)
                    new_targets = torch.concat([the_targets, pub_targets], dim = 0)
                    
                    reindexing = self.get_reindex(the_inputs.shape[0])
                    assert new_inputs.shape[0] == len(reindexing)
                    new_inputs = new_inputs[reindexing]
                    new_targets = new_targets[reindexing]
                    
                    # print(f'111new input shape: {new_inputs.shape}')
                    # print(f'111new target shape: {new_targets.shape}')
                    
                    assert new_inputs.shape[0] == the_inputs.shape[0] * (self.times_larger+1), f'new input shape: {new_inputs.shape}'
                    
                    before_group_size = new_inputs.shape[0] 
                    group_size = self.arg_setup.group_size
                    new_inputs = torch.stack(torch.split(new_inputs, group_size, dim = 0))
                    new_targets = torch.stack(torch.split(new_targets, group_size, dim = 0))
                    
                    assert new_inputs.shape[0] == before_group_size // group_size, f'new input shape: {new_inputs.shape}'
                    # print(f'new input shape: {new_inputs.shape}')
                    # print(f'new target shape: {new_targets.shape}')

                    per_grad = self.get_per_grad(new_inputs, new_targets)
                    # for p in per_grad:
                    #     print(f'p shape: {p.shape}')
                    # exit()

                    if index % 1 == 0: 
                        self.sampling_noise_summary(index, per_grad)
 
                    # # ''' per grad checking '''
                    # # for checking_num in range(self.num_groups):
                    # #     print(f'checking_num: {checking_num}')
                    # #     print(f'inputs shape: {new_inputs.shape}')
                    # #     print(f'targets shape: {new_targets.shape}')
                    # #     # start = checking_num * (self.arg_setup.samples_per_group + 1)
                    # #     # end = start + self.arg_setup.samples_per_group + 1
                        
                    # #     model_predict = self.model( new_inputs[checking_num] )
                    # #     loss = self.loss_metric( model_predict, new_targets[checking_num].flatten() )
                    # #     loss.backward()
                    # #     # assert len(self.model.parameters()) == len(per_grad)
                    # #     for p_stack, single_grad in zip(per_grad, self.model.parameters()):
                    # #         if single_grad.requires_grad:
                    # #             norm_1 = torch.norm(p_stack[checking_num])
                    # #             norm_2 = torch.norm(single_grad.grad)
                    # #             dif_norm = torch.norm(p_stack[checking_num] - single_grad.grad)
                    # #             print(f'norm_1: {norm_1}, norm_2: {norm_2}, dif_norm: {dif_norm}')
                    # #             if dif_norm > 0.1 * norm_1 or dif_norm > 0.1 * norm_2:
                    # #                 # print(f'wrong input shape {new_inputs[checking_num].unsqueeze(0).shape}')
                    # #                 # print(f'wrong target shape {new_targets[checking_num].shape}')
                    # #                 raise ValueError(f'norm_1: {norm_1}, norm_2: {norm_2}, dif_norm: {dif_norm}')
                    # #     print('checking pass-----------------')
                    # #     self.optimizer.zero_grad()
                    
                    
                    # ''' do something here to do form the real grad '''
                    self.other_routine( per_grad )
                    
                    '''update batch metrics'''
                    with torch.no_grad():
                        predictions = self.model(the_inputs)
                        loss = self.train_setups['loss_metric']( predictions, the_targets.flatten() )
                    metrics.batch_update(loss, predictions, the_targets)

                self.data_recorder.add_record('train_acc', float(metrics.__getattr__(self.record_data_type)))
                
                    
            else:
                for batch in loader:
                    inputs, targets = map(lambda x: x.to(self.train_setups['device']), batch)
                    
                    predicts = self.model(inputs)
                    loss = self.train_setups['loss_metric']( predicts, targets.flatten() )
                    
                    '''update batch metrics'''
                    metrics.batch_update(loss, predicts, targets)

                self.data_recorder.add_record('test_acc', float(metrics.__getattr__(self.record_data_type)))

        metrics.loss /= metrics.num_images
        logger.write_log(f'==> TIME for {train_or_val}: {int(time.time()-s)} secs')
        logger.write_log(f'    {train_or_val}: {self.record_data_type} = {float(metrics.__getattr__(self.record_data_type))*100:.2f}%' )
        
        return metrics  

    def clip_per_grad(self, per_grad):
        
        per_grad_norm = ( self._compute_per_grad_norm(per_grad, which_norm = self.arg_setup.which_norm) + 1e-6 )

        # self.arg_setup.C = self.arg_setup.C * math.exp( math.log(2/10) / self.arg_setup.iter_num)

        ''' clipping/normalizing '''
        multiplier = torch.clamp(self.arg_setup.C / per_grad_norm, max = 1)
        for index, p in enumerate(per_grad):
            ''' normalizing '''
            # per_grad[index] = p / self._make_broadcastable(per_grad_norm / self.arg_setup.C, p) 
            ''' clipping '''
            # print(f'p shape: {p.shape}, mutiplier shape: {multiplier.shape}')
            per_grad[index] = p * self._make_broadcastable( multiplier, p ) 
        return per_grad



    
    # def separate_clip_and_noise(self, per_grad, pub_per_grad = None):
        
    #     per_grad = self.clip_per_grad(per_grad)
        
    #     ''' flatten '''
    #     flattened_per_grad = self.flatten_to_rows(self.num_groups, per_grad)
    #     if pub_per_grad is not None:
    #         pub_flattened_grad = self.flatten_to_rows(self.pub_num, pub_per_grad)
    #         # print(f'pub_flattened_grad: {pub_flattened_grad}')
    #         avg_tmp = abs( torch.mean(pub_flattened_grad, dim = 0, keepdim=True) )
    #     else:
    #         avg_tmp = abs( torch.mean(flattened_per_grad, dim = 0, keepdim=True) )
    #     # print(111, flattened_per_grad.shape, avg_tmp.shape)
        
    #     ''''''
    #     # if isinstance(self.grad_momentum[0], int):
    #     #     avg_tmp = abs( torch.mean(flattened_per_grad, dim = 0, keepdim=True) )
    #     # else:
    #     #     avg_tmp = abs( torch.cat([p.reshape(1,-1) for p in self.grad_momentum], dim=1) )
    #     # ''''''
    #     # if len(self.un_flattened_grad) == 0: 
    #     #     avg_tmp = abs( torch.mean(flattened_per_grad, dim = 0, keepdim=True) )
    #     # else:
    #     #     avg_tmp = abs( torch.cat([p.reshape(1,-1) for p in self.un_flattened_grad], dim=1) )
        
    #     ''''''
    #     # if len(self.un_flattened_grad) == 0: 
    #     #     avg_tmp = abs( torch.mean(flattened_per_grad, dim = 0, keepdim=True) )
    #     # else:
    #     #     avg_tmp = abs(self.final_grad)
    #     # self.final_grad = torch.mean(flattened_per_grad, dim = 0, keepdim=True)
        
        
    #     avg_s,_ = torch.topk(avg_tmp, dim = 1, k = int(avg_tmp.shape[1] * self.arg_setup.sig_percent) )
    #     avg_the_one = avg_s[:,-1:]
        
    #     sig_loc = avg_tmp >= avg_the_one
    #     insig_loc = avg_tmp < avg_the_one
        
    #     sig_loc = sig_loc.repeat(self.num_groups,1)
    #     insig_loc = insig_loc.repeat(self.num_groups,1)
        
    #     sig_C = self.arg_setup.C_sig
    #     insig_C = self.arg_setup.C_insig
        
    #     # percent = 0.01
    #     # tmp = abs(flattened_per_grad)
    #     # s,_ =torch.topk(tmp, dim = 1, k = int(tmp.shape[1] * percent) )
        
    #     # the_one = s[:,-1:]
        
    #     # small_loc = tmp < the_one
    #     # big_loc = tmp >= the_one
        
    #     ''' find small l2 norm'''
    #     all_small = torch.clone(flattened_per_grad)
    #     all_small[sig_loc] = 0
    #     all_small_norm = torch.norm(all_small, dim = 1, keepdim=True) + 1e-6
        
    #     ''' find bigger l2 norm '''
    #     all_big = torch.clone(flattened_per_grad)
    #     all_big[insig_loc] = 0
    #     all_big_norm = torch.norm(all_big, dim = 1, keepdim=True) + 1e-6
        
        
    #     # if self.epoch > 0: 
    #     #     ratio = (self.arg_setup.sig_percent**0.5 * all_big_norm + (1-self.arg_setup.sig_percent)**0.5 * all_small_norm ) / self.arg_setup.C
    #     #     ratio = list(ratio.cpu().numpy())
    #     #     inpect_num = 20
    #     #     ratio = [round(float(i),2) for i in ratio[:inpect_num]]
    #     #     print(f'grad 1-{inpect_num} ratio: {ratio}')
    #     # if self.epoch > 5:
    #     #     print(f'all small norm: {all_small_norm[:10]}')
    #     #     print(f'all big norm: {all_big_norm[:10]}')
    #     #     exit()
        
    #     ''' clip separately '''
    #     # all_small = all_small / (all_small_norm + 1e-6) * insig_C
    #     # all_big = all_big / (all_big_norm + 1e-6) * sig_C
        
    #     all_small = all_small * torch.clamp(insig_C/ all_small_norm, max = 1)
    #     all_big = all_big * torch.clamp(sig_C/ all_big_norm, max = 1)
        
    #     ''' integrate '''
    #     final = (all_small + all_big).sum(dim = 0)
        
    #     sig_total = self.total_params * self.arg_setup.sig_percent
    #     insig_total = self.total_params * (1 - self.arg_setup.sig_percent)
    #     sub_part = [ insig_total**0.5 * insig_C, sig_total**0.5 * sig_C ]
        
    #     gamma = self.sigma * sum(sub_part)**0.5
    #     noise = gamma * torch.randn_like(final) 
        
    #     noise[sig_loc[0]] *= sig_C**0.5 / sig_total**0.25
    #     noise[insig_loc[0]] *= insig_C**0.5 / insig_total**0.25
        
    #     final_grad = (final + noise) / self.num_groups

    #     self.un_flattened_grad = []
    #     for index_pair, the_shape in zip(self.shape_interval, self.shape_list):
    #         self.un_flattened_grad.append( final_grad[index_pair[0]:index_pair[1]].reshape(*the_shape) )
        
    #     for p_stack, p in zip(self.un_flattened_grad, self.model.parameters()):
    #         if p.requires_grad:
    #             p.grad = p_stack
        
    # def pub_svd_clip_and_noise(self, per_grad, pub_per_grad = None):
        
    #     # per_grad = self.clip_per_grad(per_grad)
        
    #     # for index, (g_pri, g_pub) in enumerate(zip(per_grad, pub_per_grad)):
    #     #     ''' flattening '''
    #     #     g_pri_flatten = g_pri.reshape(self.num_groups, -1)
    #     #     g_pub_flatten = g_pub.reshape(self.pub_num, -1)
            
    #     #     this_layer_numel = g_pri_flatten.shape[1]
    #     #     '''svd from pub grad'''
    #     #     pub_mean = g_pub_flatten.mean(dim = 0, keepdim = True)
    #     #     pub_centered = g_pub_flatten - pub_mean
    #     #     emp_cov = torch.mm(pub_centered.T, pub_centered) / self.pub_num
    #     #     assert emp_cov.shape == (this_layer_numel, this_layer_numel)
            
    #     #     u, s, v = torch.svd(emp_cov)
    #     #     print(f'=> layer:{index}, layer shape: {g_pri.shape[1:]}')
    #     #     print(f' . u shpe: {u.shape},  s shape: {s.shape}, v shape: {v.shape}')
    #     #     print(f' . leading 10 eigenvalues: {s[:10]}')
    #     #     print(f' . tailing 10 eigenvalues: {s[-10:]}\n')
            
    #     g_pri_total_fla = self.flatten_to_rows(self.num_groups, per_grad)
    #     g_pub_total_fla = self.flatten_to_rows(self.pub_num, pub_per_grad)
        
    #     segmentation = 500
    #     norm_control = 0
        
        
    #     time_start = time.time()
        
    #     ''' for loop svd '''
    #     i = 0
    #     grad_holder = []
    #     while i * segmentation < self.total_params:
    #         real_dim = min(self.total_params - i * segmentation, segmentation)
    #         if i % 100 == 0:
    #             print(f'--{i}th segmentation, {self.total_params //segmentation - i } left', end='')
    #         sub_part_pri = g_pri_total_fla[:, i*segmentation:(i+1)*segmentation]
    #         sub_part_pub = g_pub_total_fla[:, i*segmentation:(i+1)*segmentation]
            
    #         '''svd from pub grad'''
    #         pub_mean = sub_part_pub.mean(dim = 0, keepdim = True)
    #         pub_centered = sub_part_pub - pub_mean
    #         emp_cov = torch.mm(pub_centered.T, pub_centered) / self.pub_num
    #         u, s, _ = torch.linalg.svd(emp_cov, full_matrices=True)
            
    #         assert emp_cov.shape == (real_dim, real_dim)
            
    #         # _, s, u = torch.linalg.svd(sub_part_pub, full_matrices=True)
            
            
    #         # tmp_noise_a = sum(s)**0.5 * real_dim**0.5
    #         # tmp_noise_b = sum(s**0.5)
    #         # print(f'ratio: {tmp_noise_a}, {tmp_noise_b}')
            
    #         # print(f'=> seg: {i}')
    #         # print(f'   u shpe: {u.shape},  s shape: {s.shape}, v shape: {v.shape}')
    #         # print(f'   leading 10 eigenvalues: {s[:10]}')
    #         # print(f'   tailing 10 eigenvalues: {s[-10:]}\n')
    #         # print(111, sum(u[:,0]*u[:,0]), sum(u[:,0]*u[:,2]))
            
    #         pri_projections = torch.mm(sub_part_pri, u)
    #         assert pri_projections.shape == (self.num_groups, real_dim)
            
    #         per_coor_clip_th = s.reshape(1, -1)**0.5 * 0.5
    #         assert per_coor_clip_th.shape == (1, real_dim), f'{per_coor_clip_th.shape}, {real_dim}'
    #         norm_control += float(sum(per_coor_clip_th.view(-1)**2))
    #         # per_coor_clip_th = per_coor_clip_th.repeat(self.num_groups, 1)
            
    #         clipped_pri_projections = torch.clamp(pri_projections, min = -per_coor_clip_th, max = per_coor_clip_th)
            
    #         clipped_sum = clipped_pri_projections.sum(dim = 0, keepdim = True)
    #         assert clipped_sum.shape == (1, real_dim)
            
    #         original_noise = torch.randn(1, real_dim, device = clipped_sum.device)
    #         noise_projection = torch.mm(original_noise, u)
    #         assert noise_projection.shape == (1, real_dim)
            
    #         scaled_noise = noise_projection * per_coor_clip_th
    #         assert scaled_noise.shape == (1, real_dim), f'{scaled_noise.shape}, {per_coor_clip_th.shape}'
            
    #         private_grad_project = (clipped_sum + scaled_noise) / self.num_groups
    #         assert private_grad_project.shape == (1, real_dim)
    #         private_grad_projected_back = torch.mm(private_grad_project, u.T)
            
    #         grad_holder.append(private_grad_projected_back)
    #         i += 1
        
    #     grad_holder = torch.cat(grad_holder, dim = 1).squeeze(0)
    #     assert grad_holder.shape == (self.total_params,), grad_holder.shape
        
        
    #     # ''' batched svd '''
    #     # num_batch = self.total_params // segmentation
    #     # some_left = self.total_params % segmentation
        
        
    #     # ''' the worked reshape '''
    #     # pri_batched_of_grad_for_svd = torch.stack(  [g_pri_total_fla[:, i*segmentation:(i+1)*segmentation] for i in range(num_batch)], dim = 0  )
    #     # assert pri_batched_of_grad_for_svd.shape == (num_batch, self.num_groups, segmentation)
        
    #     # pub_batched_of_grad_for_svd = torch.stack(  [g_pub_total_fla[:, i*segmentation:(i+1)*segmentation] for i in range(num_batch)], dim = 0  )
    #     # assert pub_batched_of_grad_for_svd.shape == (num_batch, self.pub_num, segmentation)
        
    #     # pri_batched_of_grad_for_svd_residule = g_pri_total_fla[:, num_batch*segmentation:]  
    #     # assert pri_batched_of_grad_for_svd_residule.shape == (self.num_groups, some_left)
        
    #     # pub_batched_of_grad_for_svd_residule = g_pub_total_fla[:, num_batch*segmentation:]
    #     # assert pub_batched_of_grad_for_svd_residule.shape == (self.pub_num, some_left)
        
    #     # pub_mean = pub_batched_of_grad_for_svd.mean(dim = 1, keepdim = True)
    #     # pub_centered = pub_batched_of_grad_for_svd - pub_mean
        
    #     # print(f'==> doing batch matric multiplication...')
    #     # emp_cov_batched = torch.bmm(pub_centered.transpose(1,2), pub_centered) / self.pub_num
    #     # print(f'emp_cov shape: {emp_cov_batched.shape}')
        
        
    #     # print(f'==> doing batched svd...vmap')
    #     # results = vmap(torch.linalg.svd)(emp_cov_batched, full_matrices=True)
    #     # print(111, type(results))
        
    #     # print(f'==> doing batched svd...pytorch')
    #     # # u, s, v = torch.linalg.svd(emp_cov_batched, full_matrices=True)
    #     # # print(f'   u shpe: {u.shape},  s shape: {s.shape}, v shape: {v.shape}')
        
        
    #     print(f'\nnorm control: {norm_control**0.5:.2f}')
    #     print(f'\n==> svd time: {time.time() - time_start:2f} secs')
        
        
        
    #     # self.un_flattened_grad = []
    #     # for index_pair, the_shape in zip(self.shape_interval, self.shape_list):
    #     #     self.un_flattened_grad.append( grad_holder[index_pair[0]:index_pair[1]].reshape(*the_shape) )
            
    #     # for p_stack, p in zip(self.un_flattened_grad, self.model.parameters()):
    #     #     if p.requires_grad:
    #     #         p.grad = p_stack
        
    #     for p in self.model.parameters():
    #         if p.requires_grad:
    #             p.grad = torch.zeros_like(p.data)
        
    def other_routine(self, per_grad):

        ''' ploting '''
        # if self.epoch > 0:
        # #     utility.grad_summary(self.epoch, per_grad)
        #     utility.per_grad_value(self.epoch, per_grad, C,  self.arg_setup.inf_th  )

        # ''' grad norm recording '''
        # all_grad = torch.cat([p.reshape(p.shape[0], -1) for p in per_grad], dim = 1)
        # for norm_choice in self.norm_choices:
        #     all_norms = torch.norm(all_grad, p = norm_choice, dim = 1)
        #     self.avg_pnorms_holder[norm_choice].append( float( all_norms.mean() ) )
        #     mean_of_inverse = torch.mean( 1 / all_norms )
        #     self.avg_inverse_pnorms_holder[norm_choice].append( float( mean_of_inverse if mean_of_inverse < 1 else 1 ) )
        
        
        # self.each_layer_grad_summary(per_grad)
        
        ''' vanilla dp-sgd '''
        per_grad = self.clip_per_grad(per_grad)
        assert len(self.iterator_check) == len(per_grad)
        for p_stack, p in zip(per_grad, self.model.parameters()):
            if p.requires_grad:
                p.grad = torch.sum(p_stack, dim = 0) 
                p.grad += self.arg_setup.C * self.sigma * torch.randn_like(p.grad) 
                p.grad /= self.num_of_groups
        
        print(self.arg_setup.C)       
        ''' gradient momentum '''     
        # # self.arg_setup.beta = self.arg_setup.beta * math.exp( math.log(1/1000) / self.arg_setup.iter_num)  
        # for index, p in enumerate(self.model.parameters()):
        #     p.grad = self.arg_setup.beta * self.grad_momentum[index] + p.grad #* (1 - self.arg_setup.beta)
        #     self.grad_momentum[index] = torch.clone(p.grad)
            
        # ''' gradient clipping '''
        # for p in self.model.parameters():
        #     if p.requires_grad:
        #         th = 1e-2
        #         p.grad = torch.clamp(p.grad, -th, th)
                    
        self.model_update()
        
    def _compute_per_grad_norm(self, iterator, which_norm = 2):
        all_grad = torch.cat([p.reshape(p.shape[0], -1) for p in iterator], dim = 1)
        per_grad_norm = torch.norm(all_grad, dim = 1, p = which_norm)
        # assert int(per_grad_norm.numel()) == self.num_groups, (int(per_grad_norm.numel()), self.num_groups)
        return per_grad_norm
    
    def _make_broadcastable(self, tensor_to_be_reshape, target_tensor):
        broadcasting_shape = (-1, *[1 for _ in target_tensor.shape[1:]])
        return tensor_to_be_reshape.reshape(broadcasting_shape)
    
    def model_update(self):
        # ''' lr scheduling '''
        # if self.epoch > 20:
        #     self.optimizer.param_groups[0]['lr'] = 0.05
        # if self.epoch > 30:
        #     self.optimizer.param_groups[0]['lr'] = 0.02
        # if self.epoch > 40:
        #     self.optimizer.param_groups[0]['lr'] = 0.01
        
        ''' update the model '''
        self.optimizer.step()
        
        # ''' copy global model to worker model'''
        # for p_worker, p_model in zip(self.worker_param_func, self.model.parameters()):
        #     assert p_worker.shape == p_model.data.shape
        #     p_worker.copy_(p_model.data)
            
        ''' copy global model to worker model'''
        for p_worker, p_model in zip(self.worker_param_func, self.model.parameters()):
            # assert p_worker.shape == p_model.data.shape
            shape_len = len(p_model.data.shape)
            p_worker.copy_( p_model.data.unsqueeze(0).repeat(self.num_of_groups, *[1 for _ in range(shape_len)]) )
            

    def flatten_to_rows(self, leading_dim, iterator):
        return torch.cat([p.reshape(leading_dim, -1) for p in iterator], dim = 1)
    
    
    def each_layer_grad_summary(self, per_grad):
        
        for index, p in enumerate(self.model.parameters()):
            print(f'model layer-{index}, epoch-{self.epoch}:\n \
                \tshape: {p.shape}') 

        for index, p in enumerate(per_grad):
            print(f'layer-{index}, epoch-{self.epoch}:\n \
                \tshape: {p.shape}\n \
                \tgrad mean: {torch.mean(p)}\n \
                \tgrad std: {torch.std(p)}\n' 
                )
        exit()
    
                    
                    



import random
class my_randcrop(T.RandomCrop):
    @staticmethod
    def get_params(img, output_size) :
        _, h, w = None, 32,32
        th, tw = output_size

        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th )
        j = random.randint(0, w - tw )
        return i, j, th, tw

class my_RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(self, img):
        if random.random() < self.p:
            return img.flip(-1)
        return img
