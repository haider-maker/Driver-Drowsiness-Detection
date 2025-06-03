import dlib
from pathlib import Path

# === CONFIG ===
training_xml_path = Path("train.xml")
output_model_path = Path("shape_predictor_ir.dat")

# === Pre-checks ===
print(f"🔍 Checking if training file exists at {training_xml_path.resolve()}")
if not training_xml_path.exists():
    print("❌ ERROR: Training XML file not found. Please check the path.")
    exit(1)
else:
    print("✅ Training XML found.")

print(f"📁 Output will be saved to: {output_model_path.resolve().parent}")
if not output_model_path.parent.exists():
    print("⚠️ Output directory does not exist. Creating it...")
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
else:
    print("✅ Output directory exists.")

# === Training Options ===
options = dlib.shape_predictor_training_options()
options.tree_depth = 4
options.nu = 0.1
options.cascade_depth = 10
options.feature_pool_size = 400
options.num_test_splits = 50
options.oversampling_amount = 5
options.oversampling_translation_jitter = 0.1
options.be_verbose = True
options.num_threads = 4

print("🚀 Starting training...")
dlib.train_shape_predictor(str(training_xml_path), str(output_model_path), options)

print(f"✅ Training complete. Model saved to {output_model_path}")
