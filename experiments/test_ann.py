from pathlib import Path
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Ann.network import NeuralNetwork
from src.Ann.builder import ANNBuilder, create_simple_regression_nn

# test basic forward propagation functionality
def test_basic_functionality():
    print("TEST 1: Basic Functionality")
    print("="*60)
    
    # Create a simple network
    architecture = [8, 9, 1]
    activations = ['relu', 'linear']
    nn = NeuralNetwork(architecture, activations, seed=42)
    
    # Display network summary
    nn.summary()
    print(f"\nTotal Parameters: {nn.get_num_parameters()}")
    
    # Test single sample
    x_single = np.random.default_rng(0).random(8)
    y_single = nn.forward(x_single)
    print(f"\nSingle sample output shape: {y_single.shape}")
    print(f"Output value: {y_single}")
    
    # Test batch
    x_batch = np.random.default_rng(1).random((5, 8))
    y_batch = nn.forward(x_batch)
    print(f"\nBatch output shape: {y_batch.shape}")
    print(f"First 3 predictions: {y_batch[:3].flatten()}")

# """Test different builder patterns."""
def test_builder_patterns():
    print("\n" + "="*60)
    print("TEST 2: ANNBuilder Patterns")
    print("="*60)
    
    # Simple regression network
    print("\n Simple Regression Network:")
    nn1 = create_simple_regression_nn(input_size=8, hidden_size=10, seed=42)
    nn1.summary()
    
    # Deep network
    print("\n Deep Regression Network:")
    nn2 = ANNBuilder.build_deep_network(
        input_size=8, 
        hidden_size=15, 
        num_hidden_layers=3,
        seed=42
    )
    nn2.summary()
    
    # Binary classification network
    print("\n Binary Classification Network:")
    nn3 = ANNBuilder.build_binary_classification_network(
        input_size=8,
        hidden_layers=[20, 10],
        seed=42
    )
    nn3.summary()
    
    # Custom network
    print("\n Custom Network Architecture:")
    nn4 = ANNBuilder.build_custom_network(
        architecture=[8, 20, 10, 1],
        activations=['tanh', 'relu', 'linear'],
        seed=42
    )
    nn4.summary()

# Testing networks with different activation functions
def test_different_activations():
    
    print("\n" + "="*60)
    print("TEST 3: Different Activation Functions")
    print("="*60)
    
    x_test = np.random.default_rng(42).random(8)
    
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


def test_batch_processing():
    print("\n" + "="*60)
    print("TEST 4: Batch Processing")
    print("="*60)
    
    nn = create_simple_regression_nn(8, 10, seed=42)
    
    # Different batch sizes
    for batch_size in [1, 10, 50, 100]:
        x_batch = np.random.default_rng(42).random((batch_size, 8))
        y_batch = nn(x_batch)
        print(f"\nBatch size {batch_size:3d}: Output shape = {y_batch.shape}")

# testing with a dataset
def test_concrete_dataset():
    print("\n" + "="*60)
    print("TEST 5: Concrete Dataset Example")
    print("="*60)
    
    try:
        import pandas as pd

         # point to the Dataset folder next to the project root
        dataset_dir = Path(__file__).resolve().parent.parent / "Dataset"
        train_path = dataset_dir / "concrete_train.csv"
        test_path = dataset_dir / "concrete_test.csv"
        
        # load pre_processed data
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
            
            # Create and test network
            nn = ANNBuilder.build_regression_network(
                input_size=8,
                hidden_layers=[10],
                seed=42
            )
            
            nn.summary()
            
            # Make predictions on a few samples
            sample_predictions = nn(X_test[:5])
            print(f"\nSample predictions (first 5):")
            for i, (pred, actual) in enumerate(zip(sample_predictions, y_test[:5])):
                print(f"  Sample {i+1}: Predicted={pred[0]:6.2f}, Actual={actual:6.2f}")
            
        except FileNotFoundError:
            print("\nConcrete dataset files not found.")
            print("Skipping dataset test...")
            
    except ImportError:
        print("\npandas not available. Skipping dataset test...")

# Running all tests 
def main():

    print("\n" + "="*60)
    print("ANN IMPLEMENTATION")
    print("="*60)
    
    test_basic_functionality()
    test_builder_patterns()
    test_different_activations()
    test_batch_processing()
    test_concrete_dataset()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")


if __name__ == "__main__":
    main()