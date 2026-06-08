from pathlib import Path

def export_js_files(root_folder, output_file):
    root = Path(root_folder)

    with open(output_file, "w", encoding="utf-8") as out:
        for file_path in root.rglob("*"):
            if file_path.suffix.lower() in [".js", ".jsx"]:
                try:
                    out.write("=" * 80 + "\n")
                    out.write(f"FILE: {file_path.relative_to(root)}\n")
                    out.write("=" * 80 + "\n\n")

                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        out.write(f.read())

                    out.write("\n\n")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

# Ví dụ sử dụng
export_js_files(
    root_folder=r"C:\workspace\temp\JudgeResearch\frontend\src",
    output_file="all_js_code.txt"
)