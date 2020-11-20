""" CNN for network augmentation """
import torch
import torch.nn as nn
from models.augment_cells import AugmentCell
from models import ops

class AugmentCNN(nn.Module):
    """ Augmented CNN model """
    def __init__(self, input_size, C_in, C, n_classes, n_layers, auxiliary, genotype,
                 stem_multiplier=3):
        """
        Args:
            input_size: size of height and width (assuming height = width)
            C_in: # of input channels
            C: # of starting model channels
        """
        super().__init__()
        self.C_in = C_in
        self.C = C
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.genotype = genotype

        C_cur = 32
        self.stem = nn.Sequential(
            nn.BatchNorm2d(C_in),
            nn.Conv2d(C_in, C_cur, 5, 2, 2, bias=False),
            nn.BatchNorm2d(C_cur),
            nn.ReLU(),
            nn.Conv2d(C_cur, C_cur, 3, 2, 1, bias=False),
            nn.BatchNorm2d(C_cur),
            nn.ReLU(),
        )

        C_pp, C_p, C_cur = C_cur, C_cur, C_cur

        self.cells = nn.ModuleList()
        reduction_p = False
        for i in range(n_layers):
            if i in [1*n_layers//6, 3*n_layers//6, 5*n_layers//6]:
                C_cur *= 2
                reduction = True
            else:
                reduction = False

            cell = AugmentCell(genotype, C_pp, C_p, C_cur, reduction_p, reduction)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * len(cell.concat)
            C_pp, C_p = C_p, C_cur_out


        self.gap = nn.Sequential(
            nn.Conv2d(C_p, 512, 3, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d(1)
        )
        self.linear = nn.Linear(512, n_classes)
    def _init_weight(self):
        for m in self.modules():   # 继承nn.Module的方法
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
    def forward(self, x):
        s0 = s1 = self.stem(x)

        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)

        out = self.gap(s1)
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        return torch.squeeze(logits)

    def drop_path_prob(self, p):
        """ Set drop path probability """
        for module in self.modules():
            if isinstance(module, ops.DropPath_):
                module.p = p
