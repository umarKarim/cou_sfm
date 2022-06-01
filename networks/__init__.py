def get_disp_network(name):
    if name == 'sfml':
        from .sfmlDispNet import DispResNet 
        return DispResNet()
    elif name == 'diffnet':
        from .diffDispNet import DispNet 
        return DispNet() 
    else:
        raise 'Unknown network name'

def get_pose_network(name):
    if name == 'sfml':
        from .sfmlPoseNet import PoseResNet 
        return PoseResNet() 
    elif name == 'diffnet':
        from .diffPoseNet import PoseNet 
        return PoseNet() 
