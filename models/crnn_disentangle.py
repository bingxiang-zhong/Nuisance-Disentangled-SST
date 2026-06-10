import torch
import torch.nn as nn
import models.module as at_module


class CRNN(nn.Module):
    """ Proposed model
    """

    def __init__(self, params, cnn_in_dim=10, cnn_dim=64, res_Phi=360):
        super(CRNN, self).__init__()
        self.cnn_in_dim = cnn_in_dim
        self.cnn_dim = cnn_dim
        self.res_Phi = res_Phi
        res_flag = False
        norm_type = params['norm_type']

        # CNN Blocks
        self.cnn = nn.Sequential(
            at_module.CausCnnBlock(cnn_in_dim, cnn_dim, kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_res=res_flag,  norm_type=norm_type),
            nn.MaxPool2d(kernel_size=(4, 1)),
            at_module.CausCnnBlock(cnn_dim, cnn_dim, kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_res=res_flag,  norm_type=norm_type),
            nn.MaxPool2d(kernel_size=(2, 1)),
            at_module.CausCnnBlock(cnn_dim, cnn_dim, kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_res=res_flag,  norm_type=norm_type),
            nn.MaxPool2d(kernel_size=(2, 1)),
            at_module.CausCnnBlock(cnn_dim, cnn_dim, kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_res=res_flag,  norm_type=norm_type),
            nn.MaxPool2d(kernel_size=(2, 1)),
            at_module.CausCnnBlock(cnn_dim, cnn_dim, kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_res=res_flag,  norm_type=norm_type),
            nn.MaxPool2d(kernel_size=(2, 5)),
        )

        ratio = 2
        rnn_in_dim = 256
        rnn_hid_dim = 256
        rnn_out_dim = 128 * 2 * ratio
        rnn_bdflag = False
        if rnn_bdflag:
            rnn_ndirection = 2
        else:
            rnn_ndirection = 1
        self.rnn_bdflag = rnn_bdflag
        self.rnn = torch.nn.GRU(input_size=rnn_in_dim, hidden_size=rnn_hid_dim, num_layers=1,
                                batch_first=True, bias=True, dropout=0.0, bidirectional=rnn_bdflag)

        self.rnn_fc = nn.Sequential(
            torch.nn.Linear(in_features=rnn_ndirection * rnn_hid_dim, out_features=rnn_out_dim),  # ,bias=False
            nn.Tanh()
        )
        self.ipd2xyz = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.ipd2xyz2 = nn.Linear(256, self.res_Phi)
        self.sigmoid = nn.Sigmoid()
    

    def forward(self, x):
        fea = x
        nb, _, nf, nt = fea.shape  # (55,4,256,1249)
        fea_cnn = self.cnn(fea)  # (nb, nch, nf, nt)

        fea_rnn_in = fea_cnn.view(nb, -1, fea_cnn.size(3))  # (nb, nch*nf,nt), nt = 1

        fea_rnn_in = fea_rnn_in.permute(0, 2, 1)  # (nb, nt, nfea)

        fea_rnn, _ = self.rnn(fea_rnn_in) #output instead of hidden state

        fea_rnn_fc = self.rnn_fc(fea_rnn)  # (nb, nt, 2nf) 66,104,256

        fea_rnn_fc_2 = self.relu(self.ipd2xyz(fea_rnn_fc))
        doa_logits = self.sigmoid(self.ipd2xyz2(fea_rnn_fc_2))
        return doa_logits  # (nb, nt, res_Phi)


        

class crnnFE(nn.Module):
    """ Feature extractor (from input to GRU output) """

    def __init__(self, params, cnn_in_dim=10, cnn_dim=64):
        super(crnnFE, self).__init__()
        self.cnn_in_dim = cnn_in_dim
        self.cnn_dim = cnn_dim
        res_flag = False
        norm_type = params['norm_type']

        # CNN Blocks
        self.cnn = nn.Sequential(
            at_module.CausCnnBlock(cnn_in_dim, cnn_dim, kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_res=res_flag, norm_type = norm_type),
            nn.MaxPool2d(kernel_size=(4, 1)),
            at_module.CausCnnBlock(cnn_dim, cnn_dim, kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_res=res_flag, norm_type = norm_type),
            nn.MaxPool2d(kernel_size=(2, 1)),
            at_module.CausCnnBlock(cnn_dim, cnn_dim, kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_res=res_flag, norm_type = norm_type),
            nn.MaxPool2d(kernel_size=(2, 1)),
            at_module.CausCnnBlock(cnn_dim, cnn_dim, kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_res=res_flag, norm_type = norm_type),
            nn.MaxPool2d(kernel_size=(2, 1)),
            at_module.CausCnnBlock(cnn_dim, cnn_dim, kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_res=res_flag, norm_type = norm_type),
            nn.MaxPool2d(kernel_size=(2, 5)),
        )
        # self.rnn = nn.GRU(
        #     input_size=256,
        #     hidden_size=256,
        #     num_layers=1,
        #     batch_first=True,
        #     bidirectional=False
        # )


    def forward(self, x):
        # CNN processing
        nb, _, nf, nt = x.shape
        fea_cnn = self.cnn(x)

        # Prepare RNN input
        fea_rnn_in = fea_cnn.view(nb, -1, fea_cnn.size(3)).permute(0, 2, 1)

        # fea_rnn_in, _ = self.rnn(fea_rnn_in)
        return fea_rnn_in  # Output shape: (nb, nt, rnn_hid_dim * num_directions)

class Disentangler(nn.Module):
    """ Disentangler (after CNN) """

    def __init__(self, rnn_in_dim=256, rnn_hid_dim=256):
        super(Disentangler, self).__init__()
        rnn_bdflag = False
        self.rnn_ndirection = 2 if rnn_bdflag else 1
        self.rnn = nn.GRU(
            input_size=rnn_in_dim,
            hidden_size=rnn_hid_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=rnn_bdflag
        )
        # self.rnn_fc = nn.Sequential(
        #     nn.Linear(rnn_hid_dim, 512),  # Simplified since we know GRU output dims
        #     nn.Tanh()
        # )

    def forward(self, x):
        fea_rnn, fea_state = self.rnn(x)
        # fea_state = fea_state.squeeze(0)
        # fea_rnn = self.rnn_fc(x)

        return fea_rnn



class Localizer(nn.Module):
    """ Localizer (after GRU) """

    def __init__(self, res_Phi=360):
        super(Localizer, self).__init__()
        self.res_Phi = res_Phi
        ratio = 2
        rnn_hid_dim = 256
        rnn_out_dim = 128 * 2 * ratio

        self.rnn_fc = nn.Sequential(
            nn.Linear(rnn_hid_dim, rnn_out_dim),  # Simplified since we know GRU output dims
            nn.Tanh()
        )
        self.ipd2xyz = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.ipd2xyz2 = nn.Linear(256, res_Phi)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x is the output from FeatureExtractor (GRU output)
        x = self.rnn_fc(x)
        x = self.relu(self.ipd2xyz(x))
        x = self.sigmoid(self.ipd2xyz2(x)) # remove sigmoid
        return x



class CLUBLoss(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size=512, logvar_min=-5.0, logvar_max=2.0):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, y_dim),
        )

        self.p_logvar = nn.Sequential(
            nn.Linear(x_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, y_dim),
        )

    def get_mu_logvar(self, x):
        mu = self.p_mu(x)
        logvar = self.p_logvar(x)

        # smooth bound to [logvar_min, logvar_max]
        logvar = torch.tanh(logvar)
        logvar = 0.5 * (self.logvar_max - self.logvar_min) * logvar + 0.5 * (self.logvar_max + self.logvar_min)
        return mu, logvar

    @staticmethod
    def log_prob_gaussian(y, mu, logvar):
        inv_var = torch.exp(-logvar)
        logp = -0.5 * (logvar + (y - mu).pow(2) * inv_var)
        return logp.sum(dim=-1)

    def forward(self, x, y):
        if x.dim() == 3 and y.dim() == 3:
            B, T, Cx = x.shape
            _, _, Cy = y.shape
            perm_b = torch.randperm(B, device=y.device)
            y_shuf_full = y[perm_b, :, :]

            x_flat = x.reshape(B * T, Cx)
            y_flat = y.reshape(B * T, Cy)
            y_shuf = y_shuf_full.reshape(B * T, Cy)

            mu, logvar = self.get_mu_logvar(x_flat)
            pos = self.log_prob_gaussian(y_flat, mu, logvar)
            neg = self.log_prob_gaussian(y_shuf, mu, logvar)
            return (pos - neg).mean()

        # fallback: (N,C)
        if x.dim() == 3:
            B, T, Cx = x.shape
            x = x.reshape(B * T, Cx)
        if y.dim() == 3:
            B, T, Cy = y.shape
            y = y.reshape(B * T, Cy)

        mu, logvar = self.get_mu_logvar(x)
        pos = self.log_prob_gaussian(y, mu, logvar)

        perm = torch.randperm(y.size(0), device=y.device)
        y_shuf = y[perm]
        neg = self.log_prob_gaussian(y_shuf, mu, logvar)

        return (pos - neg).mean()

    def learning_loss(self, x, y):
        """
        Train q(y|x) by maximizing log-likelihood of true pairs
        (equivalently minimize NLL). Typically called with x,y detached.
        """
        if x.dim() == 3:
            B, T, Cx = x.shape
            x = x.reshape(B * T, Cx)
        if y.dim() == 3:
            B, T, Cy = y.shape
            y = y.reshape(B * T, Cy)

        mu, logvar = self.get_mu_logvar(x)
        nll = -self.log_prob_gaussian(y, mu, logvar).mean()
        return nll
