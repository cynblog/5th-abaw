import torch

import torch.nn as nn
# import torch.nn.Function
# import torch.nn.fu


class Classifier(nn.Module):
    def __init__(self, input_dim, mode=None):
        super(Classifier, self).__init__()
        self.mode = mode
        self.model = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.Linear(256, 1),
                )
    def forward(self, input):
        if self.mode == 'p':
            return torch.clamp(self.model(input), min=0, max=1)
        elif self.mode == 'n':
            return torch.clamp(self.model(input), min=-1, max=0)
        else:
            return torch.clamp(self.model(input), min=-1, max=1)

class Classifier_P(nn.Module):
    def __init__(self, input_dim):
        super(Classifier_P, self).__init__()
        self.model = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.Linear(256, 5),
                )
        self.Softmax = nn.Softmax(dim=1)
    def forward(self, input):
        # print(torch.clamp(self.model(input), min=-1, max=1))
        # return self.Softmax(torch.clamp(self.model(input), min=-1, max=1))
        return torch.clamp(self.model(input), min=-1, max=1)

class Classifier_A(nn.Module):
    def __init__(self, input_dim, input_dim2=0):
        super(Classifier_A, self).__init__()
        self.fc = nn.Linear(input_dim, 512)
        self.model = nn.Sequential(
                nn.Linear(512 + input_dim2, 256),
                nn.BatchNorm1d(256),
                nn.Linear(256, 1))
    def forward(self, input, input2=None):
        features = self.fc(input)
        if input2 is not None:
            features = torch.cat([features, input2], dim=1)
        return torch.clamp(self.model(features), min=-1, max=1)

class Classifier_V(nn.Module):
    def __init__(self, input_dim):
        super(Classifier_V, self).__init__()
        self.fc = nn.Linear(input_dim, 512)
        self.model = nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.Linear(256, 1))
    def forward(self, input):
        features = self.fc(input)
        return features, torch.clamp(self.model(features), min=-1, max=1)

class Classifier_VA(nn.Module):
    def __init__(self, v_input_dim, a_input_dim, a_input_dim2=0, mode=None):
        super(Classifier_VA, self).__init__()
        self.mode = mode
        self.classifier_a = Classifier_A(a_input_dim, a_input_dim2)
        self.classifier_v = Classifier_V(v_input_dim)
    
    def forward(self, features):
        if self.mode == 'help':
            out_features, out_v = self.classifier_v(features)
            out_a = self.classifier_a(features, out_features)
        else:
            _, out_v = self.classifier_v(features)
            out_a = self.classifier_a(features)
        return out_v, out_a

if __name__ =='__main__':
    # classifier_a = Classifier_A(1000, 512).cuda()
    # classifier_v = Classifier_V(1000).cuda()
    # features = torch.rand(3, 1000).cuda()
    # out_features, out_v = classifier_v(features)
    # out_a = classifier_a(features, out_features)
    # classifier = Classifier_VA(1000, 1000, 512, mode='help').cuda()
    classifier = Classifier_P(1000).cuda()
    features = torch.rand(3, 1000).cuda()
    out = classifier(features)
    print(out)
    # print(out_a)
