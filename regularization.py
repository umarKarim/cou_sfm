import torch 
import torch.nn as nn 


class MemRegularizer():
    def __init__(self, opts, depth_model: nn.Module, pose_model: nn.Module):
        self.opts = opts 
        self.gpus = opts.gpus
        if len(self.gpus) == 0:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:' + str(self.gpus[0]))
        curr_depth_params = self.get_flat_parameters(depth_model)
        curr_pose_params = self.get_flat_parameters(pose_model)
        self.curr_params = torch.cat((curr_depth_params, curr_pose_params)).detach().clone()
        # for pipelining results 
        self.diff = torch.zeros_like(self.curr_params)
        self.all_params = torch.ones_like(self.curr_params)
        
    def mem_regularize_loss(self, depth_model: nn.Module, pose_model: nn.Module):
        depth_params = self.get_flat_parameters(depth_model)
        pose_params = self.get_flat_parameters(pose_model)
        self.all_params = torch.cat((depth_params, pose_params))
        self.diff = (self.all_params - self.curr_params).abs()
        reg_loss = torch.dot(self.all_params.detach().abs(), self.diff)
        return reg_loss 

    def update_importance(self, depth_model: nn.Module, pose_model: nn.Module):
        self.curr_params = self.all_params.clone().detach()
    
    def get_flat_parameters(self, model: nn.Module):
        flat_params = torch.cat([p.view(-1) for p in model.parameters()]).to(self.device)
        return flat_params 
    
    

if __name__ == '__main__':
    device = torch.device('cuda:0')
    opts = Options().opts 
    disp_model = DispResNet.DispResNet().to(device)
    pose_model = PoseResNet.PoseResNet().to(device)
    
    print('Models loaded')
    MemReg = MemRegularizer(opts, disp_model, pose_model)
    print('Initialization works')
    loss = MemReg.mem_regularize_loss(disp_model, pose_model) 
    print('Loss value: {}'.format(loss))

