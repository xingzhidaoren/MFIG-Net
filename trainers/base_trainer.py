import os
import sys
import glob
import torch
from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    def __init__(self, model, optimizer='Adam', lr=0.01,  device='cpu'):
        '''
        model: the model to train
        optimizer:the name of optimizer used to train model, optional: Adam, AdamW, SGD, RMSprop
        lr: the base learning rate
     
        device: the device for training 'cpu' or 'cuda'
        '''
        self.model = model.to(device)
        self.optimizer = optimizer
        self.lr = lr     
        self.device=device
        self.save_path = ''
        self.checkpoint_file = ''
        
    @abstractmethod
    def start_train(train_dataset, save_model_dir, val_dataset=None, max_ck_files_to_save=2, pretrained_file=None):
        '''
        用来被继承和重写  NotImplemented表示未实现的方法或函数
        you have to rewrite this function in your sub-class
        '''
        return NotImplemented
    
            
    def _prepare_path(self, save_model_dir):
        print('Making checkpoint path: ', save_model_dir)
        os.makedirs(save_model_dir, exist_ok=True)
            
        self.save_path = save_model_dir
        
    def _load_pretrained(self, checkpoint_file):
        print('checking pretrained checkpoint file:', checkpoint_file)
        if not os.path.exists(checkpoint_file):
            print('finding the last checkpoint file ...')
            checkpoint_file = self._find_last_checkpoint_file()
            if checkpoint_file == '':
                print('no checkpoint file found !')
                return
        
        print('loading weights from:', checkpoint_file)
        self.model.load_state_dict(torch.load(checkpoint_file, map_location=self.device), strict=False)
       
    def _find_last_checkpoint_file(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            The path of the last checkpoint file
        """
        weights_files = glob.glob(os.path.join(self.save_path, '*.pth'))
        if len(weights_files) == 0:
            return ''
        weights_files = sorted(weights_files, key=lambda x: os.path.getmtime(x))
        return weights_files[-1]

    def _delete_old_weights(self, nun_max_keep):
        '''
        keep num_max_keep weight files, the olds are deleted
        :param nun_max_keep:
        :return:
        '''
        weights_files = glob.glob(os.path.join(self.save_path, '*.pth'))    # 获得所有匹配的文件路径列表
        if len(weights_files) <= nun_max_keep:
            return

        weights_files = sorted(weights_files, key=lambda x: os.path.getmtime(x))

        weights_files = weights_files[0:len(weights_files) - nun_max_keep]

        for weight_file in weights_files:
            if weight_file != self.checkpoint_file:
                os.remove(weight_file)

    def _draw_progress_bar(self, cur, total, bar_len=50):
        cur_len = int(cur/total*bar_len)
        sys.stdout.write('\r')
        sys.stdout.write("[{:<{}}] {}/{}".format("=" * cur_len, bar_len, cur, total))
        sys.stdout.flush()
