import os

input_dir = "D:/DROZY/annotations-auto"        
output_base_dir = "./frames" 
skip_factor = 5                                 
base_filename = "frame_"

for filename in os.listdir(input_dir):
    if filename.endswith("-s2.txt"):
        input_path = os.path.join(input_dir, filename)
        
        # Extract prefix like "1-1" from "1-1-s2.txt"
        folder_name = filename.rsplit("-s", 1)[0]
        output_dir = os.path.join(output_base_dir, folder_name)
        os.makedirs(output_dir, exist_ok=True)

        with open(input_path, 'r') as f:
            lines = f.readlines()

        frame_idx = 0
        for i in range(0, len(lines), skip_factor):
            line = lines[i]
            out_filename = f"{base_filename}{frame_idx:04}.txt"
            out_path = os.path.join(output_dir, out_filename)

            with open(out_path, 'w') as out_f:
                out_f.write(line)

            frame_idx += 1

        print(f"Processed: {filename} → {folder_name}/ ({frame_idx} frames)")

print("All matching `-s2` files processed.")
