import transformers.modeling_utils as modeling_utils
import inspect

print(f"Checking 'transformers.modeling_utils' for 'GradientCheckpointingLayer'...")
if hasattr(modeling_utils, 'GradientCheckpointingLayer'):
    print("FOUND: GradientCheckpointingLayer is in transformers.modeling_utils")
else:
    print("NOT FOUND in transformers.modeling_utils")
    
print("\nScanning for anything with 'Checkpoint' in modeling_utils:")
for name in dir(modeling_utils):
    if 'Checkpoint' in name:
        print(f" - {name}")
