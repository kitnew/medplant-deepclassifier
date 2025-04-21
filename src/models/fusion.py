import torch
import torch.nn as nn
from typing import Tuple

class SerialBasedFeatureFusion(nn.Module):
    """
    Серіальна інтеграція ознак із двох джерел із кроковим відбором за ентропією.

    Підтримує різні розміри вхідних векторів k1 та k2.
    Поетапно конкатенує і відбирає top-1024 вимірів:
      1) (a, b) -> fused1
      2) (fused1, b) -> fused2
    Обидва вектори (N×1024) передаються у незалежні класифікатори.
    """
    def __init__(
        self,
        input_dims: Tuple[int, int],  # (k1, k2)
        fused_dim: int = 1024,
        num_classes: int = 10
    ):
        super(SerialBasedFeatureFusion, self).__init__()
        k1, k2 = input_dims
        self.k1 = k1
        self.k2 = k2
        self.fused_dim = fused_dim

        # Класифікатори для кожного етапу
        self.classifier1 = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(
        self,
        a: torch.Tensor,  # (N, k1)
        b: torch.Tensor   # (N, k2)
    ):
        N, _ = a.shape
        assert a.shape[1] == self.k1, f"Expected a with dim {self.k1}, got {a.shape[1]}"
        assert b.shape[1] == self.k2, f"Expected b with dim {self.k2}, got {b.shape[1]}"

        # Етап 1: конкатенуємо a та b
        S1 = torch.cat([a, b], dim=1)               # (N, k1+k2)
        fused1 = self._select_topk_by_entropy(S1)   # (N, fused_dim)

        # Етап 2: конкатенуємо fused1 та b
        S2 = torch.cat([fused1, b], dim=1)          # (N, fused_dim+k2)
        fused2 = self._select_topk_by_entropy(S2)   # (N, fused_dim)

        # Класифікація
        logits1 = self.classifier1(fused1)
        logits2 = self.classifier2(fused2)

        return logits1, logits2, fused1, fused2

    def _select_topk_by_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Відбір топ-k ознак за Shannon-ентропією по стовпцях матриці x.
        """
        # Нормалізація: абсолютні значення (негативні зводяться до позитивних)
        abs_x = torch.abs(x)                                         # (N, D)
        probs = abs_x / (abs_x.sum(dim=0, keepdim=True) + 1e-8)      # (N, D)

        # Ентропія по стовпцях
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=0)      # (D,)

        # Вибір індексів топ-k ентропій
        topk_idx = torch.topk(entropy, self.fused_dim, largest=True).indices
        return x[:, topk_idx]                                        # (N, fused_dim)
