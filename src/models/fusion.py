import torch
import torch.nn as nn

class SerialBasedFeatureFusion(nn.Module):
    def __init__(self, feature_dim: int = 1024, bins: int = 30, top_k_a: int = None, top_k_b: int = None):
        """
        :param feature_dim: кількість ознак у вхідних векторах (за замовчуванням 1024)
        :param bins: число бінів для обчислення гістограми при оцінці ентропії
        :param top_k_a: необов'язково — взяти лише top_k_a ознак із першого потоку
        :param top_k_b: необов'язково — взяти лише top_k_b ознак із другого потоку
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.bins = bins
        self.top_k_a = top_k_a
        self.top_k_b = top_k_b

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        :param a: тензор розміру [N, feature_dim] з виходом із Residual block
        :param b: тензор розміру [N, feature_dim] з виходом із Inverted Residual block
        :return: злитий тензор розміру [N, K_a + K_b] (за замовчуванням K_a=K_b=feature_dim)
        """
        # обчислення ентропії по кожному стовпцю
        H_a = self._compute_entropy(a)
        H_b = self._compute_entropy(b)

        # сортування індексів за спаданням ентропії
        idx_a = torch.argsort(H_a, descending=True)
        idx_b = torch.argsort(H_b, descending=True)

        # опційний відбір top-k ознак
        if self.top_k_a is not None:
            idx_a = idx_a[:self.top_k_a]
        if self.top_k_b is not None:
            idx_b = idx_b[:self.top_k_b]

        # селекція та конкатенація
        a_sel = a[:, idx_a]
        b_sel = b[:, idx_b]
        fused = torch.cat([a_sel, b_sel], dim=1)
        return fused

    def _compute_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Оцінка ентропії кожної ознаки за гістограмним методом:
            H = -sum(p * log(p)), p = hist_counts / sum(hist_counts)
        :param x: [N, feature_dim]
        :return: тензор [feature_dim] ентропій
        """
        N, F = x.shape
        ent = x.new_zeros(F)
        for j in range(F):
            col = x[:, j]
            mn, mx = col.min(), col.max()
            if mn == mx:
                # константна ознака дає нуль ентропії
                ent[j] = 0.0
                continue
            hist = torch.histc(col, bins=self.bins, min=mn.item(), max=mx.item())
            p = hist / hist.sum()
            ent[j] = -(p * (p + 1e-12).log()).sum()
        return ent