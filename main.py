import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print("\nWater Leak Detection System")
print("="*50)
print("\nStep 1: Generating datasets...")
import emulate_data

print("\nStep 2: Training autoencoder...")
import autoencoder_water

print("\nStep 3: Training isolation forest...")
import isolation_water

print("\n" + "="*50)
print("Complete! Check PNG files for visualizations.")
print("="*50)
