import torch
import numpy as np

class FeatureSelectionChOA:
    """
    Feature Selection using Chimp Optimization Algorithm (ChOA) in PyTorch.
    
    This class adapts the ChOA for binary feature selection problems.
    
    References:
    - Khishe, M., & Mosavi, A. (2020). Chimp Optimization Algorithm: A New Metaheuristic for Engineering Design Problems.
    """
    
    def __init__(self, X, y, fitness_fn, pop_size=30, max_iter=100, 
                 binary_method='sigmoid', device='cpu', verbose=False):
        """
        Args:
            X: Feature matrix of shape [num_samples, num_features]
            y: Target vector of shape [num_samples]
            fitness_fn: Function that evaluates the quality of selected features
                        It should accept (X_selected, y, mask) and return a scalar value
                        where lower is better (e.g., error rate, negative accuracy)
            pop_size: Number of chimpanzees (population size)
            max_iter: Maximum number of iterations
            binary_method: Method for binarization ('sigmoid' or 'threshold')
            device: 'cpu' or 'cuda'
            verbose: Whether to print progress information
        """
        self.X = X
        self.y = y
        self.fitness_fn = fitness_fn
        self.num_features = X.shape[1]
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.binary_method = binary_method
        self.device = device
        self.verbose = verbose
        
        # Initialize population in [0, 1]
        self.population = torch.rand(pop_size, self.num_features, device=device)
        
        # Evaluate initial population
        self.fitness = torch.zeros(pop_size, device=device)
        for i in range(pop_size):
            binary_mask = self._binarize(self.population[i])
            self.fitness[i] = self.fitness_fn(X, y, binary_mask)
            
        # Record global best
        best_idx = torch.argmin(self.fitness)
        self.best_pos = self.population[best_idx].clone()
        self.best_mask = self._binarize(self.best_pos)
        self.best_fitness = self.fitness[best_idx].item()
        
        # For tracking progress
        self.convergence_curve = torch.zeros(max_iter, device=device)
        
    def _sigmoid(self, x):
        """Apply sigmoid function to input"""
        return 1.0 / (1.0 + torch.exp(-x))
        
    def _binarize(self, x):
        """
        Convert continuous values to binary using the specified method
        
        Args:
            x: Tensor of continuous values
            
        Returns:
            Binary tensor (0s and 1s)
        """
        if self.binary_method == 'sigmoid':
            # Sigmoid binarization with stochastic threshold (like Binary PSO)
            probs = self._sigmoid(x)
            random_values = torch.rand_like(probs)
            return (probs > random_values).float()
        else:
            # Simple threshold binarization
            return (x > 0.5).float()
        
    def optimize(self):
        """
        Run the optimization process.
        
        Returns:
            best_features_mask: Binary tensor indicating selected features (1=selected, 0=not selected)
            X_selected: Original feature matrix with only selected features
            convergence_curve: History of best fitness values
        """
        for t in range(self.max_iter):
            # Linearly decreasing coefficient m from 2 to 0
            m = 2.0 * (1.0 - t / float(self.max_iter))
            
            # Identify the top 4 chimps: attacker, barrier, chaser, driver
            sorted_idx = torch.argsort(self.fitness)
            A_pos = self.population[sorted_idx[0]]  # attacker (best)
            B_pos = self.population[sorted_idx[1]]  # barrier (2nd best)
            C_pos = self.population[sorted_idx[2]]  # chaser (3rd best)
            D_pos = self.population[sorted_idx[3]]  # driver (4th best)
            
            # Expand to shape (pop_size, num_features) for broadcasting
            A = A_pos.unsqueeze(0).expand(self.pop_size, -1)
            B = B_pos.unsqueeze(0).expand(self.pop_size, -1)
            C = C_pos.unsqueeze(0).expand(self.pop_size, -1)
            D = D_pos.unsqueeze(0).expand(self.pop_size, -1)
            
            # Random coefficients for 4 roles
            r1 = torch.rand(self.pop_size, self.num_features, device=self.device)
            r2 = torch.rand(self.pop_size, self.num_features, device=self.device)
            r3 = torch.rand(self.pop_size, self.num_features, device=self.device)
            r4 = torch.rand(self.pop_size, self.num_features, device=self.device)
            r5 = torch.rand(self.pop_size, self.num_features, device=self.device)
            r6 = torch.rand(self.pop_size, self.num_features, device=self.device)
            r7 = torch.rand(self.pop_size, self.num_features, device=self.device)
            r8 = torch.rand(self.pop_size, self.num_features, device=self.device)
            
            # Compute role-specific parameters with different exploration/exploitation balances
            # Attacker: More exploitation (focused search)
            attacker_weight = 0.9  # High weight for exploitation
            A1 = 2 * m * r1 - m * attacker_weight
            C1 = 2 * r2 * attacker_weight
            
            # Barrier: Balanced exploration/exploitation
            barrier_weight = 0.7
            A2 = 2 * m * r3 - m * barrier_weight
            C2 = 2 * r4 * barrier_weight
            
            # Chaser: More exploration (wider search)
            chaser_weight = 0.5
            A3 = 2 * m * r5 - m * chaser_weight
            C3 = 2 * r6 * chaser_weight
            
            # Driver: Most exploration (widest search)
            driver_weight = 0.3  # Low weight for more exploration
            A4 = 2 * m * r7 - m * driver_weight
            C4 = 2 * r8 * driver_weight
            
            # Compute D vectors with role-specific strategies
            D_attacker = C1 * A - m * self.population
            D_barrier = C2 * B - m * self.population
            D_chaser = C3 * C - m * self.population
            D_driver = C4 * D - m * self.population
            
            # Compute new positions for each role
            X1 = A - A1 * D_attacker
            X2 = B - A2 * D_barrier
            X3 = C - A3 * D_chaser
            X4 = D - A4 * D_driver
            
            # Update population with weighted average based on role importance
            # Attacker has more influence in later iterations (exploitation phase)
            attacker_influence = 0.4 + 0.3 * (t / self.max_iter)
            barrier_influence = 0.3 - 0.1 * (t / self.max_iter)
            chaser_influence = 0.2 - 0.1 * (t / self.max_iter)
            driver_influence = 0.1 - 0.05 * (t / self.max_iter)
            
            # Normalize weights
            total = attacker_influence + barrier_influence + chaser_influence + driver_influence
            attacker_influence /= total
            barrier_influence /= total
            chaser_influence /= total
            driver_influence /= total
            
            # Weighted update
            self.population = (
                attacker_influence * X1 + 
                barrier_influence * X2 + 
                chaser_influence * X3 + 
                driver_influence * X4
            )
            
            # Boundary control
            self.population = torch.clamp(self.population, 0.0, 1.0)
            
            # Evaluate fitness for each solution
            for i in range(self.pop_size):
                binary_mask = self._binarize(self.population[i])
                self.fitness[i] = self.fitness_fn(self.X, self.y, binary_mask)
            
            # Update global best
            current_best_idx = torch.argmin(self.fitness)
            current_best_fitness = self.fitness[current_best_idx].item()
            
            if current_best_fitness < self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_pos = self.population[current_best_idx].clone()
                self.best_mask = self._binarize(self.best_pos)
            
            # Store convergence history
            self.convergence_curve[t] = self.best_fitness
            
            # Print progress if verbose
            if self.verbose and (t % 10 == 0 or t == self.max_iter - 1):
                selected_count = int(self.best_mask.sum().item())
                total_count = self.num_features
                print(f"Iteration {t+1}/{self.max_iter}, Best fitness: {self.best_fitness:.6f}, "
                      f"Selected features: {selected_count}/{total_count} "
                      f"({100 * selected_count / total_count:.2f}%)")
        
        # Create final binary mask and selected features
        best_features_mask = self._binarize(self.best_pos)
        X_selected = self.X[:, best_features_mask.bool()]
        
        return best_features_mask, X_selected, self.convergence_curve


# Example usage:
if __name__ == "__main__":
    # Example: Feature Selection with ChOA
    # Create synthetic dataset
    torch.manual_seed(42)
    num_samples = 1000
    num_features = 2048
    
    # Generate random data
    X = torch.rand(num_samples, num_features)
    
    # Make only the first 5 features relevant
    true_weights = torch.zeros(num_features)
    true_weights[:5] = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2])
    
    # Generate target with some noise
    y = X @ true_weights + 0.1 * torch.randn(num_samples)
    
    # Define a simple fitness function using MSE
    def feature_selection_fitness(X, y, mask):
        """
        Evaluate feature subset using Mean Squared Error.
        Lower is better.
        
        Args:
            X: Feature matrix [num_samples, num_features]
            y: Target vector [num_samples]
            mask: Binary feature mask [num_features]
        
        Returns:
            MSE value (scalar)
        """
        # If no features selected, return a high error
        if mask.sum() == 0:
            return torch.tensor(float('inf'))
        
        # Get selected features
        X_selected = X[:, mask.bool()]
        
        # Simple linear regression using pseudo-inverse
        X_with_bias = torch.cat([X_selected, torch.ones(X_selected.shape[0], 1)], dim=1)
        beta = torch.linalg.pinv(X_with_bias) @ y
        
        # Predict and compute MSE
        y_pred = X_with_bias @ beta
        mse = torch.mean((y - y_pred) ** 2)
        
        # Add regularization to prefer fewer features
        regularization = 0.01 * mask.sum() / mask.size(0)
        
        return mse + regularization
    
    # Run feature selection
    fs_optimizer = FeatureSelectionChOA(
        X=X,
        y=y,
        fitness_fn=feature_selection_fitness,
        pop_size=30,
        max_iter=50,
        binary_method='sigmoid',  # Use sigmoid binarization
        device='cpu',
        verbose=True
    )
    
    best_mask, X_selected, convergence = fs_optimizer.optimize()
    
    print("Example: Feature Selection with ChOA")
    print(f"Selected {int(best_mask.sum().item())} out of {num_features} features")
    print("Selected feature indices:", torch.nonzero(best_mask).flatten().tolist())
    print("Best fitness value:", fs_optimizer.best_fitness)
    print(f"Shape of selected features: {X_selected.shape}")
