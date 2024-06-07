# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import torch.utils.data as data
from torch.utils.data import DataLoader
from .datasets import make_dataset, make_data_loader
from libs.utils import fix_random_seed

from PIL import Image
import os
import numpy as np
from numpy.random import randint
import random
    

class QILSetTask:
    def __init__(self, cfg, set_tasks, memory_size, shuffle, train_enable = True, shuffle_task_order=False):
        
        self.memory = {}
        self.num_tasks = len(set_tasks)
        self.shuffle = shuffle
        self.current_task = 0
        self.current_task_dataset = None
        self.memory_size = memory_size
        self.set_tasks = set_tasks
        self.train_enable = train_enable
        self.cfg = cfg
        self.template_list = list(self.set_tasks)
        self.shuffle_task_order = shuffle_task_order
        if self.shuffle_task_order:
            random.shuffle(self.template_list)
        self.init_task() # MQ
    
    def init_task(self):
        print('...Adding ids...')
        idx = 0
        new_Ntasks = []
        sorted_items = sorted(self.set_tasks.items(), key=lambda item: item[0])
        self.set_tasks = {key: value for key, value in sorted_items}
        for task_idx, task in self.set_tasks.items():
            task_n = {}
            for key, class_n in task['label_dict'].items():
                for video in task['dict_db']: 
                    video['idx'] = idx
                    if class_n in video['labels']:
                        if class_n in task_n:
                            task_n[class_n].append(video)
                        else:
                            task_n[class_n] = [video]    
                        idx += 1
            new_Ntasks.append(task_n)
        print('...Replacing Data...')
        self.set_tasks = new_Ntasks
    
    def __iter__(self):
        self.memory = {}
        self.current_task_dataset = None
        self.current_task = 0
        return self
    
    def get_data(self, data, is_memory=False):
        new_data = {}
        for class_n, videos in data.items():
            new_data[class_n] = []
            for video in videos:
                idx = video['idx']
                video['is_memory'] = is_memory
                new_data[class_n].append(video)
        return new_data
    
    def __next__(self):
        data = self.set_tasks[self.current_task]
        new_data = self.get_data(data, is_memory=False)
        # data = {self.template_list[self.current_task]: self.set_tasks[self.template_list[self.current_task]]}
        if self.train_enable:
            new_mem = self.get_data(self.memory, is_memory = True)
            comp_data = {**new_mem, **new_data}
        else:
            comp_data = new_data
        
        rng_generator = fix_random_seed(self.cfg['init_rand_seed'], include_cuda=True)
        current_task_dataset = make_dataset(name=self.cfg['dataset_name'], is_training=True, split=self.cfg['train_split'], current_task_data=comp_data, **self.cfg['dataset'])
        self.current_task_dataloader = make_data_loader(current_task_dataset, True, rng_generator, **self.cfg['loader'])
        # if self.train_enable:
        #     self.rehearsal_randomMethod(data)
            
        self.current_task += 1
        if self.current_task < len(self.set_tasks):
            return data, self.current_task_dataloader, len(self.set_tasks[self.current_task].keys())
        else:
            return data, self.current_task_dataloader, None
    
    def set_memory(self, memory):
        self.memory = memory
    
    def get_valSet_by_taskNum(self, num_task):
        eval_data = {}
        total_data = []
        list_val_loaders = []
        list_num_classes = []
        for k in range(num_task):
            data = self.set_tasks[k]
            eval_data = {**eval_data, **data}
            new_data = self.get_data(data)
            total_data.append(new_data)
            list_num_classes.append(len(eval_data.keys()))
        for i, data_i in enumerate(total_data):
            self.cfg['dataset']['num_classes'] = list_num_classes[i]
            val_task_dataset = make_dataset(name=self.cfg['dataset_name'], is_training=False, split=self.cfg['val_split'], current_task_data=total_data[:i+1], **self.cfg['dataset'])
            val_task_dataloader = make_data_loader(val_task_dataset, False, None, 1, self.cfg['loader']['num_workers'])
            list_val_loaders.append((val_task_dataloader, list_num_classes[i]))
        return list_val_loaders  
    
    def rehearsal_randomMethod(self, current_task):
        saved_queries = self.memory.keys()
        current_query = current_task.keys()
        num_queries = len(saved_queries) + len(current_query)
        elem_to_save = {**self.memory, **current_task}
        if self.memory_size != 'ALL':
            num_instances_per_query= self.memory_size // num_queries
            for query_n, elems in elem_to_save.items():
                random.shuffle(elems['dict_db'])
                elem_to_save[query_n]['dict_db'] = elems['dict_db'][:num_instances_per_query]
        self.memory = elem_to_save
        
    def get_dataloader(self, data, batch_size=1, memory=None, sample_frame=False):
        is_memory = True if sample_frame else False
        data = self.get_data(data, is_memory)
        if memory != None:
            new_mem = self.get_data(memory, is_memory = True)
            data = {**new_mem, **data}
        
        rng_generator = fix_random_seed(self.cfg['init_rand_seed'], include_cuda=True)
        dataset = make_dataset(name=self.cfg['dataset_name'], is_training=True, split=self.cfg['train_split'], current_task_data=data, **self.cfg['dataset'])
        self.cfg['loader']['batch_size'] = batch_size
        dataloader = make_data_loader(dataset, True, rng_generator, **self.cfg['loader'])
        return dataloader
    
    
class BiCQILSetTask:
    def __init__(self, cfg, set_tasks, memory_size, shuffle, train_enable = True, shuffle_task_order=False):
        
        self.memory = {}
        self.num_tasks = len(set_tasks)
        self.shuffle = shuffle
        self.current_task = 0
        self.current_task_dataset = None
        self.memory_size = memory_size
        self.set_tasks = set_tasks
        self.train_enable = train_enable
        self.cfg = cfg
        self.template_list = list(self.set_tasks)
        self.shuffle_task_order = shuffle_task_order
        if self.shuffle_task_order:
            random.shuffle(self.template_list)
        self.init_task() # MQ
        self.perc = 0.9
    
    def init_task(self):
        print('...Adding ids...')
        idx = 0
        new_Ntasks = []
        sorted_items = sorted(self.set_tasks.items(), key=lambda item: item[0])
        self.set_tasks = {key: value for key, value in sorted_items}
        for task_idx, task in self.set_tasks.items():
            task_n = {}
            for key, class_n in task['label_dict'].items():
                for video in task['dict_db']: 
                    video['idx'] = idx
                    if class_n in video['labels']:
                        if class_n in task_n:
                            task_n[class_n].append(video)
                        else:
                            task_n[class_n] = [video]    
                        idx += 1
            new_Ntasks.append(task_n)
        print('...Replacing Data...')
        self.set_tasks = new_Ntasks
    
    def __iter__(self):
        self.memory = {}
        self.current_task_dataset = None
        self.current_task = 0
        return self
    
    def get_data(self, data, is_memory=False):
        new_data = {}
        for class_n, videos in data.items():
            new_data[class_n] = []
            for video in videos:
                idx = video['idx']
                video['is_memory'] = is_memory
                new_data[class_n].append(video)
        return new_data
    
    def __next__(self):
        data = self.set_tasks[self.current_task]
        new_data = self.get_data(data, is_memory=False)
        # data = {self.template_list[self.current_task]: self.set_tasks[self.template_list[self.current_task]]}
        if self.train_enable:
            new_mem = self.get_data(self.memory, is_memory = True)
            comp_data = {**new_mem, **new_data}
        else:
            comp_data = new_data
        
        rng_generator = fix_random_seed(self.cfg['init_rand_seed'], include_cuda=True)
        if self.current_task == 0:
            current_task_dataset = make_dataset(name=self.cfg['dataset_name'], is_training=True, split=self.cfg['train_split'], current_task_data=comp_data, **self.cfg['dataset'])
            self.current_task_dataloader = make_data_loader(current_task_dataset, True, rng_generator, **self.cfg['loader'])
            len_train_data = len(current_task_dataset.data_list)
            return data, self.current_task_dataloader, None, len_train_data, None, len(self.set_tasks[self.current_task].keys())
        else:
            train_train_data = {}
            train_val_data = {}
            for key, values in comp_data.items():
                total_data_value = len(values)
                len_train_train_data = int(total_data_value*self.perc)
                train_train_data[key] = values[:len_train_train_data]
                train_val_data[key] = values[len_train_train_data:]
                
            train_train_dataset = make_dataset(name=self.cfg['dataset_name'], is_training=True, split=self.cfg['train_split'], current_task_data=train_train_data, **self.cfg['dataset'])
            train_train_dataloader = make_data_loader(train_train_dataset, True, rng_generator, **self.cfg['loader'])
            len_train_data = len(train_train_dataloader.data_list)
            
            train_val_dataset = make_dataset(name=self.cfg['dataset_name'], is_training=True, split=self.cfg['train_split'], current_task_data=train_val_data, **self.cfg['dataset'])
            train_val_dataloader = make_data_loader(train_val_dataset, True, rng_generator, **self.cfg['loader'])
            len_val_data = len(train_val_dataset.data_list)
            
            if self.current_task >=1 and self.current_task < len(self.set_tasks) - 1:
                self.current_task += 1
                return data, train_train_dataloader, train_val_dataloader, len_train_data, len_val_data, len(self.set_tasks[self.current_task].keys())
            else:
                return data, train_train_dataloader, train_val_dataloader, len_train_data, len_val_data, None
    
    def set_memory(self, memory):
        self.memory = memory
    
    def get_valSet_by_taskNum(self, num_task):
        eval_data = {}
        total_data = []
        list_val_loaders = []
        list_num_classes = []
        for k in range(num_task):
            data = self.set_tasks[k]
            eval_data = {**eval_data, **data}
            new_data = self.get_data(data)
            total_data.append(new_data)
            list_num_classes.append(len(eval_data.keys()))
        for i, data_i in enumerate(total_data):
            self.cfg['dataset']['num_classes'] = list_num_classes[i]
            val_task_dataset = make_dataset(name=self.cfg['dataset_name'], is_training=False, split=self.cfg['val_split'], current_task_data=total_data[:i+1], **self.cfg['dataset'])
            val_task_dataloader = make_data_loader(val_task_dataset, False, None, 1, self.cfg['loader']['num_workers'])
            list_val_loaders.append((val_task_dataloader, list_num_classes[i]))
        return list_val_loaders  
    
    def rehearsal_randomMethod(self, current_task):
        saved_queries = self.memory.keys()
        current_query = current_task.keys()
        num_queries = len(saved_queries) + len(current_query)
        elem_to_save = {**self.memory, **current_task}
        if self.memory_size != 'ALL':
            num_instances_per_query= self.memory_size // num_queries
            for query_n, elems in elem_to_save.items():
                random.shuffle(elems['dict_db'])
                elem_to_save[query_n]['dict_db'] = elems['dict_db'][:num_instances_per_query]
        self.memory = elem_to_save
        
    def get_dataloader(self, data, batch_size=1, memory=None, sample_frame=False):
        is_memory = True if sample_frame else False
        data = self.get_data(data, is_memory)
        if memory != None:
            new_mem = self.get_data(memory, is_memory = True)
            data = {**new_mem, **data}
        
        rng_generator = fix_random_seed(self.cfg['init_rand_seed'], include_cuda=True)
        dataset = make_dataset(name=self.cfg['dataset_name'], is_training=True, split=self.cfg['train_split'], current_task_data=data, **self.cfg['dataset'])
        self.cfg['loader']['batch_size'] = batch_size
        dataloader = make_data_loader(dataset, True, rng_generator, **self.cfg['loader'])
        return dataloader