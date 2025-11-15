from pathlib import Path
import numpy as np

import sys
import os
# Allow imports from the project structure (ANN + PSO)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Ann.network import NeuralNetwork
from src.Ann.builder import ANNBuilder, create_simple_regression_nn


# TEST 1: Basic forward propagation and shape checks
def test_basic_functionality():
    print("TEST 1: Basic Functionality")
    print("="*60)
    
    # Create a simple ANN with fixed architecture for testing
    architecture = [8, 9, 1]
    activations = ['relu', 'linear']
    nn = NeuralNetwork(architecture, activations, seed=42)
    
    # Show model summary (helps verify layer sizes + activations)
    nn.summary()
    print(f"\nTotal Parameters: {nn.get_num_parameters()}")
    
    # Test forward pass on a single vector
    x_single = np.random.default_rng(0).random(8)
    y_single = nn.forward(x_single)
    print(f"\nSingle sample output shape: {y_single.shape}")
    print(f"Output value: {y_single}")
    
    # Test behaviour with batched inputs
    x_batch = np.random.default_rng(1).random((5, 8))
    y_batch = nn.forward(x_batch)
    print(f"\nBatch output shape: {y_batch.shape}")
    print(f"First 3 predictions: {y_batch[:3].flatten()}")

# TEST 2: Making sure ANNBuilder creates networks correctly
def test_builder_patterns():
    print("\n" + "="*60)
    print("TEST 2: ANNBuilder Patterns")
    print("="*60)
    
    # Simple regression model (1 hidden layer)
    print("\n Simple Regression Network:")
    nn1 = create_simple_regression_nn(input_size=8, hidden_size=10, seed=42)
    nn1.summary()
    
    # A deeper regression model
    print("\n Deep Regression Network:")
    nn2 = ANNBuilder.build_deep_network(
        input_size=8, 
        hidden_size=15, 
        num_hidden_layers=3,
        seed=42
    )
    nn2.summary()
    
    # Binary classifier (tests sigmoid output handling)
    print("\n Binary Classification Network:")
    nn3 = ANNBuilder.build_binary_classification_network(
        input_size=8,
        hidden_layers=[20, 10],
        seed=42
    )
    nn3.summary()
    
    # Fully custom architecture + activations
    print("\n Custom Network Architecture:")
    nn4 = ANNBuilder.build_custom_network(
        architecture=[8, 20, 10, 1],
        activations=['tanh', 'relu', 'linear'],
        seed=42
    )
    nn4.summary()

# TEST 3: Confirming different activation functions behave correctly
def test_different_activations():
    
    print("\n" + "="*60)
    print("TEST 3: Different Activation Functions")
    print("="*60)
    
    x_test = np.random.default_rng(42).random(8)
    
    # Try multiple activation setups to ensure forward pass consistency
    activations_to_test = [
        ['relu', 'linear'],
        ['tanh', 'linear'],
        ['sigmoid', 'linear'],
        ['relu', 'sigmoid']
    ]
    
    for act in activations_to_test:
        nn = NeuralNetwork([8, 10, 1], act, seed=42)
        output = nn(x_test)
        print(f"\nActivations {act}: Output = {output[0]:.4f}")


# TEST 4: Ensure the network handles variable batch sizes
def test_batch_processing():
    print("\n" + "="*60)
    print("TEST 4: Batch Processing")
    print("="*60)
    
    nn = create_simple_regression_nn(8, 10, seed=42)
    
    # Try a range of batch sizes to check shape handling
    for batch_size in [1, 10, 50, 100]:
        x_batch = np.random.default_rng(42).random((batch_size, 8))
        y_batch = nn(x_batch)
        print(f"\nBatch size {batch_size:3d}: Output shape = {y_batch.shape}")

# TEST 5: Concrete dataset loading + forward pass sanity check
def test_concrete_dataset():
    print("\n" + "="*60)
    print("TEST 5: Concrete Dataset Example")
    print("="*60)
    
    try:
        import pandas as pd

        # Locate the dataset folder (expected to be placed next to project root)
        dataset_dir = Path(__file__).resolve().parent.parent / "data"
        train_path = dataset_dir / "concrete_train.csv"
        test_path = dataset_dir / "concrete_test.csv"
        
        # Try loading processed dataset splits
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            X_train = train_df.iloc[:, :-1].values
            y_train = train_df.iloc[:, -1].values
            X_test = test_df.iloc[:, :-1].values
            y_test = test_df.iloc[:, -1].values
            
            print(f"\nDataset loaded successfully!")
            print(f"Training samples: {X_train.shape[0]}")
            print(f"Test samples: {X_test.shape[0]}")
            print(f"Features: {X_train.shape[1]}")
            
            # Create a simple ANN and run a few predictions
            nn = ANNBuilder.build_regression_network(
                input_size=8,
                hidden_layers=[10],
                seed=42
            )
            
            nn.summary()
            
            # Evaluate the first few samples to ensure forward pass works on real data
            sample_predictions = nn(X_test[:5])
            print(f"\nSample predictions (first 5):")
            for i, (pred, actual) in enumerate(zip(sample_predictions, y_test[:5])):
                print(f"  Sample {i+1}: Predicted={pred[0]:6.2f}, Actual={actual:6.2f}")
            
        except FileNotFoundError:
            print("\nConcrete dataset files not found.")
            print("Skipping dataset test...")
            
    except ImportError:
        print("\npandas not available. Skipping dataset test...")

# Main test runner
def main():

    print("\n" + "="*60)
    print("ANN IMPLEMENTATION")
    print("="*60)
    
    # Run all test suites
    test_basic_functionality()
    test_builder_patterns()
    test_different_activations()
    test_batch_processing()
    test_concrete_dataset()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")


if __name__ == "__main__":
    main()
