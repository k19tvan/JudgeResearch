# backend/file_manager.py

## Original Path
`backend/file_manager.py`

## Purpose
Manages the disk-level operations for the problem bank. Includes creating structure, unzipping uploaded archives, and validating testcase payload integrity on the file system.

## Imports
- `os`, `shutil`: For directory mapping and filesystem sweeping.
- `zipfile`: To safely deflate uploaded problem testcases.
- `fastapi.UploadFile`: For handling multipart form data.

## Classes
N/A.

## Functions
- `initialize_problem_storage(storage_base, name_slug)`: Provisions a folder for a new problem.
- `save_and_unzip_file(dest_parent_folder, upload, subfolder_name)`: Saves zip blobs to disk and deflates them dynamically. Validates inner `.json` content.
- `validate_folder_structure(...)`: Ensures the unzipped contents correspond to the required problem schema (matching pairs of input/output testcases).

## Execution Notes
Routinely operates on `storage/problems/{slug_name}` definitions.

## Modification Risks
Improperly securing unzipping operations or failing to sanitize `UploadFile.filename` can cause zip-slip vulnerabilities or overwrite critical system files.

## Related Files
- [src_backend_sandbox.md](src_backend_sandbox.md)
- [src_backend_main.md](src_backend_main.md)
