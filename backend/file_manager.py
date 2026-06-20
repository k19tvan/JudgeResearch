import os
import shutil
import zipfile
from fastapi import UploadFile


def initialize_problem_storage(storage_base: str, name_slug: str) -> str:
    """Khởi tạo và tạo thư mục lưu trữ cho một problem cụ thể dựa trên tên slug."""
    path = os.path.join(storage_base, "problems", name_slug)
    os.makedirs(path, exist_ok=True)
    return path


def save_and_unzip_file(dest_parent_folder: str, upload: UploadFile, subfolder_name: str) -> str:
    """Lưu tệp zip tạm thời lên đĩa, thực hiện giải nén nội dung vào thư mục đích,
    sau đó tự động xóa tệp nén zip đệm để tiết kiệm dung lượng đĩa cho server.
    """
    os.makedirs(dest_parent_folder, exist_ok=True)
    temp_zip_path = os.path.join(dest_parent_folder, f"temp_{subfolder_name}.zip")
    
    # Ghi file zip tạm thời
    with open(temp_zip_path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
        
    # Tạo thư mục giải nén chính thức (ví dụ: /inputs hoặc /outputs)
    extract_path = os.path.join(dest_parent_folder, subfolder_name)
    os.makedirs(extract_path, exist_ok=True)
    
    # Thực hiện giải nén toàn bộ
    extract_root = os.path.abspath(extract_path)
    with zipfile.ZipFile(temp_zip_path, 'r') as z:
        for member in z.infolist():
            member_path = os.path.abspath(os.path.join(extract_root, member.filename))
            if member_path != extract_root and not member_path.startswith(extract_root + os.sep):
                raise ValueError("Unsafe zip archive path")
        z.extractall(extract_path)
        
    # Xóa file zip đệm tạm thời
    if os.path.exists(temp_zip_path):
        os.remove(temp_zip_path)
        
    return extract_path


def validate_folder_structure(folder_path: str) -> bool:
    """Đảm bảo thư mục giải nén tồn tại và có chứa ít nhất một tệp tin testcase .json hợp lệ."""
    if not os.path.exists(folder_path):
        return False
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.json')]
    return len(files) > 0
