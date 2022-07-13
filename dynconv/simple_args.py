

class SimpleArguments:
    def __init__(self):
        self.data_root = ""
        self.pretrain_weight = ""
        self.load_weight = ""

    def update_model_args(self, idx, args):
        if idx == "hardware_resnet50":
            args.model = "resnet50"
            args.model_cfg = "hardware"
            args.mask_type = "conv"
            args.mask_kernel = 1
            args.no_attention = True
        elif idx == "hardware_2048":
            args.model = "resnet50"
            args.model_cfg = "hardware_2048"
            args.mask_type = "conv"
            args.mask_kernel = 1
            args.no_attention = True
        elif idx == "default_mask":
            args.mask_type = "conv"
            args.mask_kernel = 1
            args.no_attention = True
        elif not idx:
            pass
        else:
            raise NotImplementedError
        return args

    def update_loss_args(self, idx, args):
        if idx == "layer_wise":
            args.net_weight = 0
            args.valid_range = 1
        elif idx == "flops":
            args.valid_range = 0.33
        elif not idx:
            pass
        else:
            raise NotImplementedError
        return args

    def update(self, args):
        if args.model_args:
            print("Using simple model args: {}".format(args.model_args))
            args = self.update_model_args(args.model_args, args)
        if args.loss_args:
            print("Using loss model args: {}".format(args.loss_args))
            args = self.update_loss_args(args.loss_args, args)
        if self.data_root:
            args.dataset_root = self.data_root

        if not args.evaluate:
            if args.load == "pretrain":
                args.load = self.pretrain_weight
            elif self.load_weight and args.load == "default":
                print("Replacing weight checkpoint: {} -> {}".format(args.load, self.load_weight))
                args.load = self.load_weight
        return args
