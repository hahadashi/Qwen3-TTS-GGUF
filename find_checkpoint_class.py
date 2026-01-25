import os
import transformers

package_path = os.path.dirname(transformers.__file__)
print(f"Transformers package path: {package_path}")

target_str = "class GradientCheckpointing"
print(f"Searching for '{target_str}' in {package_path}...")

found = False
for root, dirs, files in os.walk(package_path):
    for file in files:
        if file.endswith(".py"):
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line_idx, line in enumerate(f):
                        if target_str in line:
                            print(f"\nFOUND in: {file_path}")
                            print(f"Line {line_idx+1}: {line.strip()}")
                            found = True
            except Exception as e:
                pass

if not found:
    print("\n'GradientCheckpointing' class definition NOT found in transformers package.")
