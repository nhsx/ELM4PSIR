from typing import Tuple

import torch
from torch import nn


class Attention(nn.Module):
    """
    Attention network

    Parameters
    ----------
    rnn_size : int
        Size of Bi-LSTM
    """

    def __init__(self, rnn_size: int) -> None:
        super(Attention, self).__init__()
        self.w = nn.Linear(rnn_size, 1, bias=True)  # generally set bias to True
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        H : torch.Tensor (batch_size, word_pad_len, hidden_size)
            Output of Bi-LSTM

        Returns
        -------
        r : torch.Tensor (batch_size, rnn_size)
            Sentence representation

        alpha : torch.Tensor (batch_size, word_pad_len)
            Attention weights
        """
        # eq.9: M = tanh(H)
        M = self.tanh(H)  # (batch_size, word_pad_len, rnn_size)
        # ========== just some maths proofing =====================
        # print(f"Shape of M inside attention is: {M.shape}")
        # print(f"Shape of weights inside self.w is: {self.w.weight.shape}")
        # expanded_weights = self.w.weight.t().expand(H.shape[0], -1, -1)
        # print(f"Expanded weights shape: {expanded_weights.shape}")
        # bmm_alpha = torch.bmm(M, expanded_weights)
        # print(f"bmm alpha shape: {bmm_alpha.shape}")

        # print(f"Shape of data after self.W: {(self.w(M)).shape}")

        # raw_alpha = M @ self.w.weight.t()

        # w_array = self.w.weight.cpu().detach().numpy()
        # m_array = M.cpu().detach().numpy()
        # print(f"Shape of w array is: {w_array.shape} and m_array: {m_array.shape}")
        # raw_alpha = np.dot(w_array.T, m_array)

        # print(f"Raw alpha shape: {raw_alpha.shape}")
        # ========================= end of maths proofing ===================

        # eq.10: α = softmax(w^T M)
        alpha = self.w(M).squeeze(2)  # (batch_size, word_pad_len)
        # print(f"Shape of alpha after squeeze: {alpha.shape}")

        # print(
        #   (
        #       "abs difference between alpha and raw alpha: "
        #       f"{(raw_alpha.squeeze(2)-alpha).abs().sum()}"
        #   )
        # )
        # print(
        #   (
        #       "abs difference between alpha and bmm alpha : "
        #       f"{(bmm_alpha.squeeze(2)-alpha).abs().sum()}")
        #   )
        # )
        # print(
        #   (
        #       "abs difference between raw alpha and bmm alpha : "
        #       f"{(bmm_alpha.squeeze(2)-raw_alpha.squeeze(2)).abs().sum()}"
        #   )
        # )

        alpha = self.softmax(alpha)  # (batch_size, word_pad_len)
        # print(f"Shape of H is : {H.shape}")
        # print(f"Shape of H permuted is: {(H.transpose(1,2)).shape}")
        # print(f"shape of alpha unsqueeze is: {alpha.unsqueeze(2).shape} ")

        # eq.11: r = H * alpha.T

        # one way by third party author - #
        r = H * alpha.unsqueeze(2)  # (batch_size, word_pad_len, rnn_size)

        r = r.sum(dim=1)  # (batch_size, rnn_size)
        # print(f"shape of r after sum: {r.shape}")
        ######################################

        # another way to calculate r - from original authors of paper but results in
        # difference numbers than using the elementwise multiplcation above

        r_bmm = torch.bmm(
            H.transpose(1, 2), alpha.unsqueeze(2)
        )  # (batch_size, rnn_size, 1)
        # print(f"r bmm shape is : {r_bmm.shape}")
        r_bmm_squeeze = r_bmm.squeeze(dim=-1)  # (batch_size, rnn_size)
        # print(f"r bmm squeezed shape: {r_bmm_squeeze.shape}")
        # print(f"Shape of r before sum: {r.shape}")

        # print(f"abs difference between r and r_bmm: {(r-r_bmm_squeeze).abs().sum()}")
        return r_bmm_squeeze, alpha
