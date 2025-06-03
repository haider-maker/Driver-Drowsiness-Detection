import os

def print_top_level_structure(root_dir):
    try:
        items = sorted(os.listdir(root_dir))
        for i, item in enumerate(items):
            connector = '└── ' if i == len(items) - 1 else '├── '
            print(connector + item)
    except FileNotFoundError:
        print(f"Error: Directory '{root_dir}' not found.")
    except PermissionError:
        print(f"Error: Permission denied for directory '{root_dir}'.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Print top-level folder structure.')
    parser.add_argument('directory', type=str, help='Root directory to list')
    args = parser.parse_args()

    print(args.directory)
    print_top_level_structure(args.directory)
