import json
import subprocess
import tempfile
import os
import re
from typing import Any, Dict, List, Tuple


def parse_txt_file(file_path: str) -> Any:
    """Đọc và phân tích cú pháp tệp tin testcase .txt thô thành cấu trúc dữ liệu Python.
    - Nếu dòng chứa nhiều số phân tách bởi khoảng trắng, dòng đó trở thành một mảng (list).
    - Nếu dòng chỉ chứa một giá trị, dòng đó được làm phẳng thành biến đơn (scalar).
    - Tự động nhận diện và ép kiểu sang số nguyên (int) hoặc số thực (float).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    parsed_lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            # Tách các phần tử trên dòng dựa trên khoảng trắng
            tokens = stripped.split()
            line_values = []
            for token in tokens:
                try:
                    # Thử ép kiểu số nguyên
                    line_values.append(int(token))
                except ValueError:
                    try:
                        # Thử ép kiểu số thực
                        line_values.append(float(token))
                    except ValueError:
                        # Giữ nguyên kiểu chuỗi ký tự nếu không phải số
                        line_values.append(token)
            
            # Làm phẳng mảng nếu dòng chỉ chứa duy nhất một phần tử
            if len(line_values) == 1:
                parsed_lines.append(line_values[0])
            else:
                parsed_lines.append(line_values)
                
    # Nếu tệp tin chỉ chứa duy nhất một dòng và một phần tử, trả về phần tử gốc
    if len(parsed_lines) == 1:
        return parsed_lines[0]
    return parsed_lines


def extract_testcases_from_folders(input_folder_path: str, output_folder_path: str) -> List[Tuple[Any, Any]]:
    """Đọc toàn bộ các tệp tin .txt trong thư mục inputs và outputs tương ứng.
    Sắp xếp tự nhiên theo thứ tự số học và ghép cặp các testcase đầu vào/đầu ra.
    """
    if not os.path.exists(input_folder_path):
        raise FileNotFoundError(f"Input folder not found: {input_folder_path}")
    if not os.path.exists(output_folder_path):
        raise FileNotFoundError(f"Output folder not found: {output_folder_path}")

    def get_valid_txt_files(folder: str) -> List[str]:
        files = []
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if os.path.isdir(path):
                continue
            # Bỏ qua các file ẩn/file hệ thống của macOS hoặc Windows (.DS_Store, ._filename,...)
            if name.startswith('.'):
                continue
            # Bỏ qua thư mục rác hệ thống macOS sinh ra khi nén/giải nén
            if name.startswith('__MACOSX') or '/__MACOSX' in name:
                continue
            # Chỉ nhận tệp tin kết thúc bằng .txt
            if not name.lower().endswith('.txt'):
                continue
            files.append(name)
            
        # Sử dụng biểu thức chính quy để sắp xếp tự nhiên theo chỉ số số học (e.g. input_2 trước input_10)
        def natural_sort_key(s):
            numbers = re.findall(r'\d+', s)
            return [int(n) for n in numbers] if numbers else s
            
        return sorted(files, key=natural_sort_key)

    input_files = get_valid_txt_files(input_folder_path)
    output_files = get_valid_txt_files(output_folder_path)

    paired = []
    for i in range(min(len(input_files), len(output_files))):
        inp_path = os.path.join(input_folder_path, input_files[i])
        out_path = os.path.join(output_folder_path, output_files[i])
        paired.append((parse_txt_file(inp_path), parse_txt_file(out_path)))
    return paired


def execute_user_code(user_code: str, test_input: Any, timeout: int = 3) -> dict:
    """Thực thi mã nguồn của người dùng trong một tiến trình con (subprocess) an toàn.
    - Hỗ trợ cơ chế bóc tách tham số tự động cho định dạng 'N M' (Bounding Box, ML/DL).
    - Giải quyết triệt để lỗi NameError khi dữ liệu chứa true/false/null bằng chuỗi JSON bảo mật.
    """
    # Tuần tự hóa JSON hai lần để bọc chuỗi an toàn, ngăn chặn lỗi cú pháp rỗng/boolean trong Python con
    serialized_input_json = json.dumps(json.dumps(test_input))

    wrapper = f"""
{user_code}
import json
input_data = json.loads({serialized_input_json})

try:
    # Cơ chế tự động phát hiện tham số (Dynamic Unpacking Engine)
    if isinstance(input_data, list) and len(input_data) > 0:
        first_line = input_data[0]
        # Phát hiện định dạng 'N M' ở dòng đầu tiên để tự động phân rã thành boxes1 và boxes2
        if isinstance(first_line, list) and len(first_line) == 2 and all(isinstance(x, int) for x in first_line):
            n, m = first_line[0], first_line[1]
            if len(input_data) == 1 + n + m:
                boxes1 = input_data[1 : 1 + n]
                boxes2 = input_data[1 + n : 1 + n + m]
                res = solution(boxes1, boxes2)
            else:
                res = solution(input_data)
        else:
            res = solution(input_data)
    else:
        res = solution(input_data)

    if hasattr(res, 'tolist'):
        res = res.tolist()
    print(json.dumps({{"result": res}}))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
"""

    try:
        proc = subprocess.run(
            ["python", "-c", wrapper],
            capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return {"error": "Time Limit Exceeded"}

    if proc.returncode != 0 and not proc.stdout:
        return {"error": proc.stderr.strip()}

    out = proc.stdout.strip() or proc.stderr.strip()
    try:
        return json.loads(out)
    except Exception:
        return {"error": f"Invalid execution output: {out}"}


def compare_outputs(user_out: Any, expected_out: Any, tolerance: float = 1e-4) -> Tuple[bool, str]:
    """So sánh kết quả chạy của người dùng với kết quả mong đợi.
    - Hỗ trợ duyệt so sánh đệ quy đối với các ma trận/mảng nhiều chiều sâu [1.1.1].
    - Cho phép sai số số thực nhỏ (tolerance) đối với các kết quả tính toán Deep Learning [1.1.1].
    """
    try:
        if isinstance(expected_out, dict) and 'result' in expected_out:
            expected = expected_out['result']
        else:
            expected = expected_out

        if isinstance(user_out, dict) and 'result' in user_out:
            user = user_out['result']
        else:
            user = user_out

        # Kiểm tra và so sánh mảng đa chiều đệ quy
        if isinstance(expected, list) and isinstance(user, list):
            if len(expected) != len(user):
                return False, f'Length mismatch: expected {len(expected)}, got {len(user)}'
            for i, (u, e) in enumerate(zip(user, expected)):
                # Gọi đệ quy sâu xuống nếu phần tử bên trong tiếp tục là mảng lồng nhau
                if isinstance(e, list) and isinstance(u, list):
                    ok, msg = compare_outputs(u, e, tolerance)
                    if not ok:
                        return False, f'Error at nested list {i}: {msg}'
                else:
                    try:
                        if abs(float(u) - float(e)) > tolerance:
                            return False, f'Value mismatch at index {i}: expected {e}, got {u}'
                    except Exception:
                        if u != e:
                            return False, f'Value mismatch at index {i}: expected {e}, got {u}'
            return True, 'OK'

        # So sánh các giá trị đơn lẻ (scalar) dạng số
        try:
            if abs(float(user) - float(expected)) <= tolerance:
                return True, 'OK'
        except Exception:
            pass

        if user == expected:
            return True, 'OK'
        return False, f'Expected {expected}, got {user}'
    except Exception as e:
        return False, f'Compare error: {str(e)}'