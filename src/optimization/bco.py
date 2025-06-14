import torch
import numpy as np

class FeatureSelectionChOA:
    """
    Binary Chimp Optimization for feature selection.

    Args:
        num_features (int): Number of input features (e.g., 2048).
        pop_size (int): Population size (# chimps).
        max_iter (int): Max iterations for optimization.
        threshold (float): Sigmoid threshold for binary mask.
        device (torch.device): CPU or GPU.
    """
    def __init__(self, num_features, pop_size=20, max_iter=50, threshold=0.5, device=None):
        self.num_features = num_features
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.threshold = threshold
        self.device = device or torch.device('cpu')
        # initialize continuous positions
        self.population = torch.rand((pop_size, num_features), device=self.device)

    def _binary_mask(self, positions):
        # sigmoid + threshold
        probs = torch.sigmoid(positions)
        return (probs > self.threshold).float()

    def _evaluate(self, features, labels, masks, clf):
        with torch.no_grad():
            # Evaluate accuracy of mask on whole dataset
            fitness = torch.zeros(self.pop_size, device=self.device)
            X_np = features
            y_np = labels
            for i in range(self.pop_size):
                sel = masks[i].bool().cpu().numpy()
                if sel.sum() == 0:
                    fitness[i] = 0.0
                else:
                    clf.fit(X_np[:, sel], y_np)
                    fitness[i] = clf.score(X_np[:, sel], y_np)
        return fitness

    def optimize(self, features, labels, classifier):
        """
        Optimize feature subset mask.
        Returns:
            best_mask (1D Tensor of size num_features).
        """
        with torch.no_grad():
            for t in range(self.max_iter):
                print(f"Iteration {t+1}/{self.max_iter}")
                masks = self._binary_mask(self.population)
                fitness = self._evaluate(features, labels, masks, classifier)
                # select top4
                topk = torch.topk(fitness, 4).indices
                elites = [self.population[i].clone() for i in topk]
                a_coeff = 2 - 2 * (t / self.max_iter)
                # update positions
                for i in range(self.pop_size):
                    x = self.population[i]
                    new_positions = []
                    for idx in range(4):
                        r1, r2 = torch.rand(self.num_features, device=self.device), torch.rand(self.num_features, device=self.device)
                        A = 2 * a_coeff * r1 - a_coeff
                        C = 2 * r2
                        m = torch.rand(self.num_features, device=self.device)
                        D = C * elites[idx] - m * x
                        new_positions.append(elites[idx] - A * D)
                    self.population[i] = torch.stack(new_positions).mean(dim=0)
        # final mask
        final_masks = self._binary_mask(self.population)
        final_fitness = self._evaluate(features, labels, final_masks, classifier)
        best_idx = torch.argmax(final_fitness)
        return final_masks[best_idx].bool()

if __name__ == '__main__':
    import torch
    from models.fusion import SerialBasedFeatureFusion
    from sklearn.neighbors import KNeighborsClassifier as WNN

    # Create random input tensors
    N = 32  # batch size
    feature_dim = 1024
    a = torch.randn(N, feature_dim)
    b = torch.randn(N, feature_dim)

    top_k_a = 512
    top_k_b = 512

    # Initialize fusion module
    fusion = SerialBasedFeatureFusion(feature_dim=feature_dim, top_k_a=top_k_a, top_k_b=top_k_b)

    # Apply fusion
    fused = fusion(a, b)

    # Check output shape
    assert fused.shape == (N, top_k_a + top_k_b), f"Expected shape {(N, top_k_a + top_k_b)}, got {fused.shape}"
    print(f"Fusion successful. Output shape: {fused.shape}")
    
    # Initialize BCO
    bco = FeatureSelectionChOA(
        num_features=top_k_a + top_k_b,
        pop_size=100,
        max_iter=5,
        threshold=0.5,
        device=torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    )

    labels = torch.randint(0, 2, (N,))

    wnn = WNN()
    wnn.fit(a, labels)
    
    # Apply BCO
    mask = bco.optimize(fused, labels, wnn)
    print(f"BCO successful. Selected features: {mask}")
    print(f"Fused features with mask: {fused[:, mask]}")
    print(f"Shape of fused features with mask: {fused[:, mask].shape}")