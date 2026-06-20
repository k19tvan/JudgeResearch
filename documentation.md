# ĐẶC TẢ USE CASE HỆ THỐNG JUDGERESEARCH

## 1. Giới thiệu

Tài liệu này mô tả các use case chính của hệ thống JudgeResearch sau khi đã rà soát lại mã nguồn hiện tại. Nội dung tập trung vào hành vi nghiệp vụ, tác nhân, điều kiện, luồng xử lý và các API liên quan.

JudgeResearch là nền tảng hỗ trợ học tập lập trình, tạo bài tập, chấm bài tự động, nghiên cứu repository bằng AI, quản lý lộ trình học tập, blog cộng đồng và ticket hỗ trợ.

## 2. Tác nhân hệ thống

| Tác nhân | Mô tả |
| --- | --- |
| Khách chưa đăng nhập | Người truy cập chưa có phiên đăng nhập, chỉ có thể đăng ký/đăng nhập và xem một số nội dung công khai nếu được phép. |
| User | Người học thông thường, có thể xem bài công khai, làm bài, nộp bài, bình luận, bình chọn và tạo ticket. |
| Contributor | Người biên soạn nội dung, có thể tạo bài tập/lộ trình riêng tư, gửi yêu cầu công khai và đề xuất cập nhật lời giải. |
| Admin | Quản trị viên, có quyền quản lý tài khoản, phê duyệt nội dung, xử lý ticket và quản trị dữ liệu hệ thống. |
| Hệ thống AI | Thành phần gọi Gemini/DeepWiki để phân tích repository, sinh nội dung bài học, testcase và tài liệu nghiên cứu. |

## 3. Danh sách use case

| Mã | Tên use case | Tác nhân chính |
| --- | --- | --- |
| UC-01 | Đăng ký tài khoản | Khách chưa đăng nhập |
| UC-02 | Đăng nhập hệ thống | Khách chưa đăng nhập |
| UC-03 | Làm mới phiên đăng nhập và đăng xuất | User/Contributor/Admin |
| UC-04 | Xem và cập nhật hồ sơ cá nhân | User/Contributor/Admin |
| UC-05 | Vô hiệu hóa/xóa tài khoản cá nhân | User/Contributor/Admin |
| UC-06 | Quản lý tài khoản người dùng | Admin |
| UC-07 | Tạo bài tập thủ công | Contributor/Admin |
| UC-08 | Xem và lọc danh sách bài tập | User/Contributor/Admin |
| UC-09 | Xem nội dung bài tập | User/Contributor/Admin |
| UC-10 | Chạy thử code | User/Contributor/Admin |
| UC-11 | Nộp bài và lưu submission | User/Contributor/Admin |
| UC-12 | Xem lịch sử submission | User/Contributor/Admin |
| UC-13 | Sửa hoặc xóa bài tập | Contributor/Admin |
| UC-14 | Yêu cầu công khai và phê duyệt bài tập | Contributor/Admin |
| UC-15 | Tạo danh sách bài tập từ repository | Contributor/Admin |
| UC-16 | Quản lý draft session | Contributor/Admin |
| UC-17 | Tạo và xem roadmap | User/Contributor/Admin |
| UC-18 | Sinh nội dung chi tiết cho bước roadmap | Contributor/Admin |
| UC-19 | Lưu bước roadmap thành bài tập | Contributor/Admin |
| UC-20 | Phê duyệt, công khai và xóa roadmap | Contributor/Admin |
| UC-21 | Quản lý blog | Contributor/Admin |
| UC-22 | Bình luận và trả lời bình luận | User/Contributor/Admin |
| UC-23 | Bình chọn blog hoặc bình luận | User/Contributor/Admin |
| UC-24 | Tạo và quản lý ticket hỗ trợ | User/Contributor/Admin |
| UC-25 | Quản lý đề xuất cập nhật lời giải | Contributor/Admin |
| UC-26 | Nghiên cứu repository bằng DeepWiki | User/Contributor/Admin |

---

## UC-01. Đăng ký tài khoản

**Tác nhân chính:** Khách chưa đăng nhập.

**Mục tiêu:** Cho phép người dùng mới tạo tài khoản với vai trò mặc định là `user`.

**Tiền điều kiện:**

- Người dùng chưa đăng nhập.
- Hệ thống backend và database hoạt động.

**Hậu điều kiện:**

- Nếu thành công, một bản ghi mới được thêm vào bảng `users` với `role = 'user'` và `status = 'active'`.
- Nếu thất bại, không có tài khoản mới được tạo.

**Luồng chính:**

1. Người dùng mở trang đăng ký.
2. Người dùng nhập display name, email, username và password.
3. Người dùng nhấn nút đăng ký.
4. Frontend gửi yêu cầu `POST /api/auth/register`.
5. Backend kiểm tra dữ liệu bắt buộc, định dạng email, username, password và trùng lặp.
6. Backend băm mật khẩu bằng hàm hash bảo mật.
7. Backend lưu tài khoản mới vào bảng `users`.
8. Hệ thống trả kết quả thành công và frontend điều hướng người dùng về trang đăng nhập.

**Luồng thay thế/ngoại lệ:**

- Nếu thiếu trường bắt buộc, hệ thống hiển thị lỗi nhập liệu.
- Nếu username hoặc email đã tồn tại, backend trả lỗi tương ứng.
- Nếu password không đạt quy tắc bảo mật, backend từ chối đăng ký.

**API liên quan:** `POST /api/auth/register`.

---

## UC-02. Đăng nhập hệ thống

**Tác nhân chính:** Khách chưa đăng nhập.

**Mục tiêu:** Xác thực người dùng và cấp access token để truy cập các chức năng phù hợp với vai trò.

**Tiền điều kiện:**

- Người dùng đã có tài khoản.
- Tài khoản đang ở trạng thái `active`.

**Hậu điều kiện:**

- Nếu thành công, frontend lưu access token, user id, username, role và avatar URL nếu có.
- Nếu thất bại, người dùng vẫn ở màn hình đăng nhập.

**Luồng chính:**

1. Người dùng mở trang đăng nhập.
2. Người dùng nhập username và password.
3. Frontend gửi `POST /api/auth/login`.
4. Backend tìm tài khoản trong bảng `users`.
5. Backend kiểm tra trạng thái tài khoản.
6. Backend xác thực password với hash đã lưu.
7. Backend tạo JWT access token và refresh token.
8. Frontend lưu thông tin phiên và điều hướng vào dashboard.

**Luồng thay thế/ngoại lệ:**

- Nếu sai username/password, hệ thống trả lỗi xác thực.
- Nếu tài khoản bị `disabled`, backend từ chối đăng nhập.
- Nếu thiếu thông tin đăng nhập, frontend không gửi request hoặc backend trả lỗi.

**API liên quan:** `POST /api/auth/login`.

---

## UC-03. Làm mới phiên đăng nhập và đăng xuất

**Tác nhân chính:** User, Contributor, Admin.

**Mục tiêu:** Duy trì phiên đăng nhập hợp lệ và cho phép người dùng thoát phiên.

**Tiền điều kiện:**

- Người dùng đã đăng nhập.
- Refresh token còn hiệu lực nếu cần làm mới access token.

**Hậu điều kiện:**

- Nếu refresh thành công, frontend nhận access token mới.
- Nếu logout thành công, token phiên bị thu hồi và frontend xóa dữ liệu local.

**Luồng chính - Refresh token:**

1. Một request được gửi lên backend nhưng access token hết hạn.
2. Frontend gọi `POST /api/auth/refresh`.
3. Backend kiểm tra refresh token.
4. Backend phát hành access token mới.
5. Frontend retry request ban đầu.

**Luồng chính - Logout:**

1. Người dùng chọn đăng xuất.
2. Frontend gọi `POST /api/auth/logout`.
3. Backend thu hồi refresh token.
4. Frontend xóa thông tin phiên khỏi localStorage và quay về trang đăng nhập.

**Luồng thay thế/ngoại lệ:**

- Nếu refresh token hết hạn hoặc bị thu hồi, hệ thống yêu cầu đăng nhập lại.
- Nếu tài khoản bị disabled, refresh token không được chấp nhận.

**API liên quan:** `POST /api/auth/refresh`, `POST /api/auth/logout`.

---

## UC-04. Xem và cập nhật hồ sơ cá nhân

**Tác nhân chính:** User, Contributor, Admin.

**Mục tiêu:** Cho phép người dùng xem và cập nhật thông tin cá nhân của chính mình.

**Tiền điều kiện:**

- Người dùng đã đăng nhập.
- Request có JWT hợp lệ.

**Hậu điều kiện:**

- Nếu thành công, bảng `users` được cập nhật.
- Nếu thất bại, dữ liệu cũ được giữ nguyên.

**Luồng chính:**

1. Người dùng mở tab Profile.
2. Frontend gọi API lấy hồ sơ.
3. Người dùng chọn Edit Profile.
4. Người dùng cập nhật username, display name, email, password hoặc avatar.
5. Frontend gửi `PUT /api/users/{user_id}`.
6. Backend xác thực JWT và kiểm tra user trong token có phải chủ tài khoản hay không.
7. Backend kiểm tra định dạng và trùng lặp dữ liệu.
8. Backend lưu thay đổi và trả kết quả thành công.

**Luồng thay thế/ngoại lệ:**

- Nếu user trong JWT không khớp tài khoản cần sửa, backend trả `403`.
- Nếu email/username trùng, backend trả lỗi.
- Nếu avatar không phải JPEG/PNG/WEBP hoặc vượt dung lượng, backend từ chối.
- Nếu access token hết hạn, frontend thử refresh token.

**API liên quan:** `GET /api/users/profile/{user_id}`, `GET /api/users/{user_id}`, `PUT /api/users/{user_id}`.

---

## UC-05. Vô hiệu hóa/xóa tài khoản cá nhân

**Tác nhân chính:** User, Contributor, Admin.

**Mục tiêu:** Cho phép người dùng tự xóa tài khoản và dữ liệu liên quan của mình.

**Tiền điều kiện:**

- Người dùng đã đăng nhập.
- Người dùng xác nhận thao tác xóa.

**Hậu điều kiện:**

- Tài khoản và dữ liệu liên quan bị xóa theo logic backend.
- Frontend xóa phiên đăng nhập.

**Luồng chính:**

1. Người dùng mở Profile.
2. Người dùng chọn Deactivate Account.
3. Frontend hiển thị xác nhận.
4. Người dùng xác nhận.
5. Frontend gọi `POST /api/users/{user_id}/deactivate`.
6. Backend xác thực chủ tài khoản qua JWT.
7. Backend xóa refresh token, submission, draft session, roadmap, problem do user tạo và bản ghi user.
8. Frontend quay về trang đăng nhập.

**Luồng thay thế/ngoại lệ:**

- Nếu người dùng hủy xác nhận, hệ thống không làm gì.
- Nếu JWT không hợp lệ hoặc không phải chủ tài khoản, backend từ chối.

**API liên quan:** `POST /api/users/{user_id}/deactivate`.

---

## UC-06. Quản lý tài khoản người dùng

**Tác nhân chính:** Admin.

**Mục tiêu:** Cho phép admin xem, tìm kiếm, cập nhật thông tin, vai trò và trạng thái tài khoản.

**Tiền điều kiện:**

- Người thực hiện đã đăng nhập với role `admin`.
- JWT hợp lệ.

**Hậu điều kiện:**

- Nếu thành công, thông tin tài khoản mục tiêu được cập nhật trong bảng `users`.
- Nếu thất bại, không có thay đổi nào được lưu.

**Luồng chính:**

1. Admin mở tab Account Management hoặc Users.
2. Frontend gọi `GET /api/admin/users`.
3. Backend xác thực JWT admin.
4. Backend trả danh sách người dùng.
5. Admin chọn một người dùng.
6. Frontend gọi `GET /api/admin/users/{user_id}`.
7. Admin chỉnh display name, email, role hoặc status.
8. Frontend gửi `PUT /api/admin/users/{user_id}`.
9. Backend kiểm tra quyền admin và dữ liệu hợp lệ.
10. Backend cập nhật bảng `users`.
11. Frontend hiển thị kết quả thành công.

**Luồng thay thế/ngoại lệ:**

- Nếu người thực hiện không phải admin, backend trả lỗi phân quyền.
- Nếu role/status không hợp lệ, backend từ chối.
- Nếu email trùng hoặc sai định dạng, backend trả lỗi.

**API liên quan:** `GET /api/admin/users`, `GET /api/admin/users/{user_id}`, `PUT /api/admin/users/{user_id}`, `POST /api/users/make-contributor`, `POST /api/users/make-admin`.

---

## UC-07. Tạo bài tập thủ công

**Tác nhân chính:** Contributor, Admin.

**Mục tiêu:** Tạo bài tập lập trình mới từ nội dung nhập thủ công và testcase.

**Tiền điều kiện:**

- Người dùng đã đăng nhập.
- Người dùng có role `contributor` hoặc `admin`.

**Hậu điều kiện:**

- Bài tập mới được lưu trong bảng `problems`.
- File nội dung và testcase được lưu trong `storage/problems`.

**Luồng chính:**

1. Contributor/Admin mở tab Problems.
2. Người dùng chọn Create Problem.
3. Người dùng nhập statement, theory, tutorial, solution, coding template, checker.
4. Người dùng cung cấp testcase bằng JSON trực tiếp hoặc upload ZIP input/output.
5. Frontend gửi `POST /api/problems/create/manual` dạng `FormData`.
6. Backend xác thực JWT và role.
7. Backend tạo thư mục lưu trữ bài tập.
8. Backend lưu file markdown/code và testcase.
9. Backend tạo bản ghi trong bảng `problems`.
10. Frontend tải lại danh sách bài tập.

**Luồng thay thế/ngoại lệ:**

- Nếu thiếu dữ liệu bắt buộc, backend trả lỗi.
- Nếu ZIP không hợp lệ hoặc có đường dẫn nguy hiểm, backend từ chối giải nén.
- Nếu người dùng không đủ quyền, backend trả `403`.

**API liên quan:** `POST /api/problems/create/manual`.

---

## UC-08. Xem và lọc danh sách bài tập

**Tác nhân chính:** User, Contributor, Admin.

**Mục tiêu:** Cho phép người dùng xem bài công khai và các bài riêng tư phù hợp quyền.

**Tiền điều kiện:**

- Với filter `public`: không bắt buộc đăng nhập.
- Với filter `private` hoặc `all`: yêu cầu JWT hợp lệ.

**Hậu điều kiện:** Danh sách bài tập được hiển thị theo quyền truy cập.

**Luồng chính:**

1. Người dùng mở tab Problems hoặc Home.
2. Frontend gọi `GET /api/problems/filter`.
3. Backend xác định filter mode.
4. Backend kiểm tra JWT nếu filter yêu cầu đăng nhập.
5. Backend truy vấn bảng `problems`, kèm điểm submission tốt nhất nếu có user.
6. Frontend render danh sách bài.

**Luồng thay thế/ngoại lệ:**

- Nếu người dùng chưa đăng nhập mà xem private/all, backend trả `401`.
- Nếu user thường cố xem bài riêng tư của người khác, backend trả `403`.

**API liên quan:** `GET /api/problems/filter`.

---

## UC-09. Xem nội dung bài tập

**Tác nhân chính:** User, Contributor, Admin.

**Mục tiêu:** Hiển thị nội dung bài tập trong màn hình live coding.

**Tiền điều kiện:**

- Bài tập tồn tại.
- Người dùng có quyền xem bài tập.

**Hậu điều kiện:** Nội dung bài tập được hiển thị; lời giải/checker được ẩn hoặc hiện theo quyền.

**Luồng chính:**

1. Người dùng chọn một bài tập.
2. Frontend điều hướng đến `/livecoding/{problem_id}`.
3. Frontend gọi `GET /api/problems/{problem_id}/content`.
4. Backend kiểm tra trạng thái public/private và quyền của người dùng.
5. Backend đọc các file nội dung từ storage.
6. Backend trả statement, theory, tutorial, coding template và các phần được phép xem.

**Luồng thay thế/ngoại lệ:**

- Nếu bài riêng tư và người dùng không phải chủ sở hữu/admin, backend trả `403`.
- Nếu user chưa accepted bài, lời giải mẫu bị thay bằng thông báo hạn chế truy cập.
- Nếu bài không tồn tại, backend trả `404`.

**API liên quan:** `GET /api/problems/{problem_id}/content`.

---

## UC-10. Chạy thử code

**Tác nhân chính:** User, Contributor, Admin.

**Mục tiêu:** Cho phép người dùng chạy code trên testcase trước khi nộp chính thức.

**Tiền điều kiện:**

- Bài tập tồn tại.
- Người dùng đang ở màn hình live coding.

**Hậu điều kiện:** Kết quả chạy thử được hiển thị nhưng không nhất thiết tạo submission chính thức.

**Luồng chính:**

1. Người dùng nhập code.
2. Người dùng nhấn Run.
3. Frontend gửi `POST /api/problems/{problem_id}/run`.
4. Backend tải testcase.
5. Backend chạy code và so sánh output.
6. Backend trả trạng thái từng testcase.
7. Frontend hiển thị kết quả.

**Luồng thay thế/ngoại lệ:**

- Nếu code lỗi runtime, hệ thống trả thông báo lỗi.
- Nếu quá thời gian, testcase được đánh dấu time limit exceeded.
- Nếu testcase thiếu hoặc lỗi cấu trúc, backend trả lỗi.

**API liên quan:** `POST /api/problems/{problem_id}/run`.

---

## UC-11. Nộp bài và lưu submission

**Tác nhân chính:** User, Contributor, Admin.

**Mục tiêu:** Chấm bài chính thức và lưu kết quả vào lịch sử submission.

**Tiền điều kiện:**

- Người dùng đã đăng nhập.
- Bài tập tồn tại và người dùng có quyền truy cập.

**Hậu điều kiện:**

- Một bản ghi mới được lưu vào bảng `submissions`.
- Kết quả từng testcase có thể được lưu vào `test_runs`.

**Luồng chính:**

1. Người dùng nhập code và nhấn Submit.
2. Frontend gửi `POST /api/problems/{problem_id}/submit` kèm JWT.
3. Backend lấy `user_id` từ JWT.
4. Backend chạy toàn bộ testcase.
5. Backend tính score và trạng thái tổng.
6. Backend lưu submission.
7. Frontend hiển thị điểm và trạng thái.

**Luồng thay thế/ngoại lệ:**

- Nếu chưa đăng nhập, backend trả `401`.
- Nếu code lỗi runtime hoặc sai output, trạng thái submission phản ánh lỗi.
- Nếu hệ thống chấm gặp lỗi, backend trả thông báo lỗi phù hợp.

**API liên quan:** `POST /api/problems/{problem_id}/submit`.

---

## UC-12. Xem lịch sử submission

**Tác nhân chính:** User, Contributor, Admin.

**Mục tiêu:** Cho phép người dùng xem các lần nộp bài và chi tiết kết quả.

**Tiền điều kiện:** Người dùng đã đăng nhập.

**Hậu điều kiện:** Danh sách hoặc chi tiết submission được hiển thị.

**Luồng chính:**

1. Người dùng mở tab Submissions hoặc phần lịch sử trong live coding.
2. Frontend gọi API submission.
3. Backend xác thực JWT.
4. Backend truy vấn các submission phù hợp.
5. Frontend hiển thị trạng thái, điểm, thời gian nộp và chi tiết nếu có.

**Luồng thay thế/ngoại lệ:**

- Nếu người dùng cố xem submission không thuộc quyền của mình, backend từ chối.
- Nếu không có submission, frontend hiển thị danh sách rỗng.

**API liên quan:** `GET /api/submissions`, `GET /api/problems/{problem_id}/submissions`, `GET /api/users/{user_id}/submissions`.

---

## UC-13. Sửa hoặc xóa bài tập

**Tác nhân chính:** Contributor, Admin.

**Mục tiêu:** Cập nhật hoặc xóa bài tập đã tạo.

**Tiền điều kiện:**

- Người dùng đã đăng nhập.
- Contributor là tác giả bài tập hoặc người dùng là admin.

**Hậu điều kiện:**

- Nếu sửa, dữ liệu bài tập và file liên quan được cập nhật.
- Nếu xóa, bài tập và dữ liệu liên quan được xóa theo logic backend.

**Luồng chính - Sửa:**

1. Người dùng chọn Edit Problem.
2. Frontend tải nội dung và testcase.
3. Người dùng cập nhật dữ liệu.
4. Frontend gửi `PUT /api/problems/{problem_id}`.
5. Backend kiểm tra quyền owner/admin.
6. Backend cập nhật dữ liệu.

**Luồng chính - Xóa:**

1. Người dùng chọn Delete Problem.
2. Frontend yêu cầu xác nhận.
3. Frontend gửi `DELETE /api/problems/{problem_id}`.
4. Backend kiểm tra quyền.
5. Backend xóa bài.

**Luồng thay thế/ngoại lệ:**

- Nếu contributor không phải tác giả, backend trả `403`.
- Nếu bài không tồn tại, backend trả `404`.

**API liên quan:** `GET /api/problems/{problem_id}/testcases`, `PUT /api/problems/{problem_id}`, `DELETE /api/problems/{problem_id}`.

---

## UC-14. Yêu cầu công khai và phê duyệt bài tập

**Tác nhân chính:** Contributor, Admin.

**Mục tiêu:** Đảm bảo bài tập của contributor được admin duyệt trước khi công khai.

**Tiền điều kiện:**

- Contributor đã tạo bài riêng tư.
- Admin đã đăng nhập để phê duyệt.

**Hậu điều kiện:**

- Bài được chuyển sang public nếu được duyệt.
- Bài vẫn private nếu bị từ chối hoặc bị thu hồi.

**Luồng chính:**

1. Contributor chọn yêu cầu công khai bài.
2. Frontend gọi `POST /api/problems/{problem_id}/request-approval`.
3. Backend kiểm tra tác giả và chuyển `request_status = 'PENDING'`.
4. Admin mở Admin Queue.
5. Frontend gọi `GET /api/problems/pending-requests`.
6. Admin chọn Approve hoặc Reject.
7. Backend cập nhật `is_public` và `request_status`.

**Luồng thay thế/ngoại lệ:**

- Nếu người yêu cầu không phải tác giả/admin, backend trả `403`.
- Nếu admin từ chối, bài quay về trạng thái private.
- Admin có thể chuyển bài public/private trực tiếp theo quyền quản trị.

**API liên quan:** `POST /api/problems/{problem_id}/request-approval`, `GET /api/problems/pending-requests`, `POST /api/problems/{problem_id}/approve`, `POST /api/problems/{problem_id}/reject`, `POST /api/problems/{problem_id}/private`, `POST /api/problems/{problem_id}/public`.

---

## UC-15. Tạo danh sách bài tập từ repository

**Tác nhân chính:** Contributor, Admin.

**Mục tiêu:** Dùng AI để phân tích repository và đề xuất danh sách bài tập/lộ trình.

**Tiền điều kiện:**

- Người dùng đã đăng nhập với role phù hợp.
- Repository URL hợp lệ.
- `GEMINI_API_KEY` được cấu hình nếu gọi chức năng Gemini.

**Hậu điều kiện:** Một draft session được tạo để người dùng xem và chỉnh sửa.

**Luồng chính:**

1. Người dùng mở tab Research.
2. Người dùng nhập repository URL, mức độ, framework, ghi chú.
3. Frontend gửi `POST /api/problems/problems_from_repo`.
4. Backend gọi AI để phân tích repository và sinh danh sách đề xuất.
5. Backend lưu draft session trong bảng `draft_problem_sessions`.
6. Frontend hiển thị danh sách đề xuất.

**Luồng thay thế/ngoại lệ:**

- Nếu thiếu API key AI, backend trả lỗi cấu hình.
- Nếu AI trả dữ liệu không hợp lệ, backend cố sửa JSON hoặc trả lỗi.
- Nếu repository không truy cập được, hệ thống báo lỗi.

**API liên quan:** `POST /api/problems/problems_from_repo`.

---

## UC-16. Quản lý draft session

**Tác nhân chính:** Contributor, Admin.

**Mục tiêu:** Cho phép người dùng xem, cập nhật phản hồi, finalize hoặc xóa draft session.

**Tiền điều kiện:** Người dùng đã đăng nhập và có quyền với draft session.

**Hậu điều kiện:** Draft session được cập nhật, xóa hoặc chuyển thành roadmap.

**Luồng chính:**

1. Người dùng mở danh sách draft session.
2. Frontend gọi `GET /api/problems/draft_sessions`.
3. Người dùng chọn một draft.
4. Frontend gọi `GET /api/problems/draft_sessions/{session_id}`.
5. Người dùng chỉnh feedback hoặc nội dung đề xuất.
6. Frontend gửi cập nhật feedback hoặc finalize.
7. Backend lưu thay đổi hoặc tạo roadmap.

**Luồng thay thế/ngoại lệ:**

- Nếu user không phải chủ draft và không phải admin, backend trả `403`.
- Nếu draft không tồn tại, backend trả `404`.

**API liên quan:** `GET /api/problems/draft_sessions`, `GET /api/problems/draft_sessions/{session_id}`, `PUT /api/problems/draft_sessions/{session_id}`, `POST /api/problems/draft_sessions/feedback`, `POST /api/problems/draft_sessions/finalize`, `DELETE /api/problems/draft_sessions/{session_id}`.

---

## UC-17. Tạo và xem roadmap

**Tác nhân chính:** User, Contributor, Admin.

**Mục tiêu:** Tổ chức bài học theo lộ trình và hiển thị roadmap theo quyền truy cập.

**Tiền điều kiện:**

- Roadmap tồn tại hoặc được tạo từ draft session.
- Người dùng có quyền xem roadmap.

**Hậu điều kiện:** Roadmap và các bước liên quan được hiển thị.

**Luồng chính:**

1. Người dùng mở tab Research/Roadmaps.
2. Frontend gọi `GET /api/roadmaps`.
3. Backend lọc roadmap theo trạng thái và quyền.
4. Người dùng chọn roadmap.
5. Frontend gọi `GET /api/roadmaps/{roadmap_id}`.
6. Hệ thống hiển thị timeline các bước học.

**Luồng thay thế/ngoại lệ:**

- User thường chỉ xem roadmap public.
- Chủ sở hữu hoặc admin xem được roadmap draft/pending.
- Nếu không có quyền, backend trả `403`.

**API liên quan:** `GET /api/roadmaps`, `GET /api/roadmaps/{roadmap_id}`.

---

## UC-18. Sinh nội dung chi tiết cho bước roadmap

**Tác nhân chính:** Contributor, Admin.

**Mục tiêu:** Sinh nội dung đầy đủ cho một bước học bằng AI.

**Tiền điều kiện:**

- Roadmap step tồn tại.
- Người dùng là chủ roadmap hoặc admin.
- AI key được cấu hình nếu chức năng cần Gemini.

**Hậu điều kiện:** Bước học có draft materials và trạng thái `generated` hoặc `failed`.

**Luồng chính:**

1. Người dùng chọn Generate/Create Detailedly cho một bước.
2. Frontend gọi `POST /api/roadmap-steps/{step_id}/create_detailedly`.
3. Backend lấy thông tin step, roadmap và repository.
4. Backend gọi AI để sinh nội dung.
5. Backend tạo file draft gồm statement, theory, tutorial, solution, coding, checker và testcase.
6. Backend cập nhật trạng thái step.
7. Frontend hiển thị trạng thái hoàn tất.

**Luồng thay thế/ngoại lệ:**

- Nếu AI lỗi, step chuyển sang `failed` và lưu `error_message`.
- Nếu người dùng không có quyền, backend trả `403`.

**API liên quan:** `POST /api/roadmap-steps/{step_id}/create_detailedly`, `POST /api/roadmaps/{roadmap_id}/generate_all`, `GET /api/roadmap-steps/{step_id}/preview`.

---

## UC-19. Lưu bước roadmap thành bài tập

**Tác nhân chính:** Contributor, Admin.

**Mục tiêu:** Chuyển draft materials của một bước roadmap thành bài tập chính thức.

**Tiền điều kiện:**

- Step đang ở trạng thái `generated`.
- Draft materials tồn tại trên storage.
- Người dùng là chủ roadmap hoặc admin.

**Hậu điều kiện:**

- Một bài tập mới được tạo trong bảng `problems`.
- Step được cập nhật `problem_id` và trạng thái `saved`.

**Luồng chính:**

1. Người dùng chọn Save to Problem.
2. Frontend gọi `POST /api/roadmap-steps/{step_id}/save_to_problem`.
3. Backend kiểm tra quyền và trạng thái step.
4. Backend chuyển file draft sang khu vực lưu trữ bài tập chính thức.
5. Backend tạo bản ghi problem.
6. Backend cập nhật roadmap step.
7. Frontend cập nhật timeline và hiển thị nút Go.

**Luồng thay thế/ngoại lệ:**

- Nếu tên bài trùng, backend trả lỗi.
- Nếu thiếu file draft, backend trả lỗi.
- Nếu step chưa generated, backend từ chối lưu.

**API liên quan:** `POST /api/roadmap-steps/{step_id}/save_to_problem`.

---

## UC-20. Phê duyệt, công khai và xóa roadmap

**Tác nhân chính:** Contributor, Admin.

**Mục tiêu:** Quản lý trạng thái công khai của roadmap và xóa roadmap khi cần.

**Tiền điều kiện:**

- Roadmap tồn tại.
- Người dùng có quyền sở hữu hoặc quyền admin.

**Hậu điều kiện:** Roadmap được chuyển trạng thái hoặc bị xóa.

**Luồng chính - Phê duyệt:**

1. Contributor gửi yêu cầu công khai roadmap.
2. Backend chuyển roadmap sang `pending`.
3. Admin mở hàng đợi phê duyệt.
4. Admin approve hoặc reject.
5. Backend cập nhật trạng thái `public` hoặc `draft`.

**Luồng chính - Xóa:**

1. Người dùng có quyền chọn Delete Roadmap.
2. Frontend gửi `DELETE /api/roadmaps/{roadmap_id}`.
3. Backend kiểm tra quyền.
4. Backend xóa roadmap và các step liên quan.
5. Backend dọn dẹp draft chưa lưu nếu có.

**Luồng thay thế/ngoại lệ:**

- Nếu không phải chủ roadmap/admin, backend trả `403`.
- Admin có thể publish/unpublish trực tiếp.

**API liên quan:** `POST /api/roadmaps/{roadmap_id}/request-approval`, `POST /api/roadmaps/{roadmap_id}/approve`, `POST /api/roadmaps/{roadmap_id}/reject`, `POST /api/roadmaps/{roadmap_id}/publish`, `POST /api/roadmaps/{roadmap_id}/unpublish`, `DELETE /api/roadmaps/{roadmap_id}`.

---

## UC-21. Quản lý blog

**Tác nhân chính:** Contributor, Admin.

**Mục tiêu:** Cho phép tạo, xem, sửa, xóa bài blog cộng đồng.

**Tiền điều kiện:**

- Người tạo/sửa/xóa đã đăng nhập.
- Người tạo blog có role phù hợp.

**Hậu điều kiện:** Blog được tạo/cập nhật/xóa trong bảng `blogs`.

**Luồng chính:**

1. Người dùng mở tab Blogs.
2. Frontend gọi `GET /api/blogs`.
3. Contributor/Admin tạo bài mới.
4. Frontend gửi `POST /api/blogs`.
5. Backend lấy author từ JWT và lưu blog.
6. Chủ bài hoặc admin có thể sửa/xóa blog.

**Luồng thay thế/ngoại lệ:**

- Nếu user thường tạo blog, backend từ chối.
- Nếu người sửa/xóa không phải tác giả/admin, backend trả `403`.
- Nếu thiếu title/content, backend trả lỗi.

**API liên quan:** `GET /api/blogs`, `GET /api/blogs/{blog_id}`, `POST /api/blogs`, `PUT /api/blogs/{blog_id}`, `DELETE /api/blogs/{blog_id}`, `POST /api/blogs/upload-image`.

---

## UC-22. Bình luận và trả lời bình luận

**Tác nhân chính:** User, Contributor, Admin.

**Mục tiêu:** Cho phép thảo luận dưới blog hoặc bài tập.

**Tiền điều kiện:** Người dùng đã đăng nhập khi tạo/sửa/xóa bình luận.

**Hậu điều kiện:** Bình luận được lưu, cập nhật hoặc xóa trong bảng `comments`.

**Luồng chính:**

1. Người dùng mở blog hoặc bài tập.
2. Frontend gọi `GET /api/comments`.
3. Người dùng nhập bình luận hoặc trả lời.
4. Frontend gửi `POST /api/comments`.
5. Backend lấy user từ JWT và lưu bình luận.
6. Frontend tải lại danh sách bình luận.

**Luồng thay thế/ngoại lệ:**

- Nếu nội dung rỗng, backend từ chối.
- Người tạo bình luận được sửa bình luận của mình.
- Người tạo hoặc admin được xóa bình luận.

**API liên quan:** `GET /api/comments`, `POST /api/comments`, `PUT /api/comments/{comment_id}`, `POST /api/comments/{comment_id}/delete`.

---

## UC-23. Bình chọn blog hoặc bình luận

**Tác nhân chính:** User, Contributor, Admin.

**Mục tiêu:** Cho phép người dùng upvote/downvote blog hoặc bình luận.

**Tiền điều kiện:** Người dùng đã đăng nhập.

**Hậu điều kiện:** Bảng `votes` được cập nhật.

**Luồng chính:**

1. Người dùng bấm upvote hoặc downvote.
2. Frontend gửi `POST /api/votes`.
3. Backend lấy user từ JWT.
4. Backend kiểm tra vote hiện có.
5. Nếu vote cùng loại đã tồn tại, backend hủy vote.
6. Nếu vote ngược loại, backend cập nhật.
7. Nếu chưa vote, backend tạo vote mới.

**Luồng thay thế/ngoại lệ:**

- Nếu không cung cấp `blog_id` hoặc `comment_id`, backend trả lỗi.
- Nếu chưa đăng nhập, backend trả `401`.

**API liên quan:** `POST /api/votes`.

---

## UC-24. Tạo và quản lý ticket hỗ trợ

**Tác nhân chính:** User, Contributor, Admin.

**Mục tiêu:** Cho phép người dùng tạo ticket, đính kèm ảnh, phản hồi và theo dõi trạng thái xử lý.

**Tiền điều kiện:** Người dùng đã đăng nhập.

**Hậu điều kiện:** Ticket hoặc phản hồi được lưu trong database và ảnh được lưu trong `storage/tickets`.

**Luồng chính - Tạo ticket:**

1. Người dùng mở tab Tickets.
2. Người dùng nhập title, description và chọn ảnh nếu có.
3. Frontend gửi `POST /api/tickets`.
4. Backend lấy user từ JWT.
5. Backend kiểm tra và lưu ảnh.
6. Backend tạo ticket trạng thái `open`.

**Luồng chính - Phản hồi ticket:**

1. Người dùng mở chi tiết ticket.
2. Người dùng nhập reply và ảnh nếu có.
3. Frontend gửi `POST /api/tickets/{ticket_id}/replies`.
4. Backend lưu phản hồi.

**Luồng chính - Cập nhật trạng thái:**

1. Admin chọn Mark as Resolved hoặc Re-open.
2. Frontend gửi `POST /api/tickets/{ticket_id}/status`.
3. Backend xác thực admin và cập nhật trạng thái.

**Luồng thay thế/ngoại lệ:**

- Nếu ticket đã resolved, hệ thống có thể chặn chỉnh sửa/phản hồi theo logic hiện hành.
- Người tạo hoặc admin có quyền sửa/xóa ticket/reply.
- Chỉ admin được đổi trạng thái ticket.
- Ảnh sai định dạng hoặc vượt dung lượng bị từ chối.

**API liên quan:** `POST /api/tickets`, `GET /api/tickets`, `GET /api/tickets/{ticket_id}`, `PUT /api/tickets/{ticket_id}`, `DELETE /api/tickets/{ticket_id}`, `POST /api/tickets/{ticket_id}/replies`, `PUT /api/tickets/replies/{reply_id}`, `DELETE /api/tickets/replies/{reply_id}`, `POST /api/tickets/{ticket_id}/status`.

---

## UC-25. Quản lý đề xuất cập nhật lời giải

**Tác nhân chính:** Contributor, Admin.

**Mục tiêu:** Cho phép contributor đề xuất cập nhật lời giải mẫu và admin phê duyệt.

**Tiền điều kiện:**

- Contributor đã đăng nhập.
- Bài tập tồn tại.
- Admin đăng nhập để duyệt.

**Hậu điều kiện:** Lời giải mẫu được cập nhật nếu đề xuất được duyệt.

**Luồng chính:**

1. Contributor mở bài tập và gửi proposal lời giải.
2. Frontend gọi `POST /api/problems/{problem_id}/solution-proposal`.
3. Backend lưu proposal với trạng thái `PENDING`.
4. Admin mở danh sách proposal.
5. Frontend gọi `GET /api/admin/solution-proposals`.
6. Admin approve hoặc reject.
7. Nếu approve, backend cập nhật file lời giải của bài tập.

**Luồng thay thế/ngoại lệ:**

- Nếu contributor không đủ quyền, backend từ chối.
- Nếu admin reject, proposal chuyển trạng thái rejected và lời giải không đổi.
- Admin có thể trực tiếp cập nhật/xóa solution bằng endpoint quản trị.

**API liên quan:** `PUT /api/problems/{problem_id}/solution`, `DELETE /api/problems/{problem_id}/solution`, `POST /api/problems/{problem_id}/solution-proposal`, `GET /api/admin/solution-proposals`, `POST /api/admin/solution-proposals/{proposal_id}/action`.

---

## UC-26. Nghiên cứu repository bằng DeepWiki

**Tác nhân chính:** User, Contributor, Admin.

**Mục tiêu:** Cho phép người dùng nhập repository URL để hệ thống DeepWiki phân tích và sinh tài liệu nghiên cứu.

**Tiền điều kiện:**

- Dịch vụ DeepWiki đang chạy.
- Repository URL hợp lệ và có thể truy cập.
- Model/LLM provider của DeepWiki được cấu hình.

**Hậu điều kiện:** Người dùng xem được tài liệu phân tích repository, cấu trúc thư mục, markdown và sơ đồ nếu có.

**Luồng chính:**

1. Người dùng mở tab Wiki.
2. Người dùng nhập Git repository URL.
3. Frontend gửi yêu cầu đến dịch vụ DeepWiki.
4. DeepWiki clone repository vào vùng tạm.
5. DeepWiki phân tích file, chunk nội dung và tạo embedding.
6. DeepWiki dùng RAG để sinh tài liệu.
7. Frontend nhận kết quả qua API/WebSocket và hiển thị.

**Luồng thay thế/ngoại lệ:**

- Nếu repository không truy cập được, hệ thống hiển thị lỗi.
- Nếu DeepWiki chưa chạy, frontend không nhận được kết quả.
- Nếu model provider lỗi, quá trình sinh tài liệu thất bại.

**Thành phần liên quan:** `deepwiki-open/api`, `WikiTab.jsx`, `websocket_wiki.py`, `rag.py`.

---

## 4. Ghi chú bảo mật và vận hành

1. Các API nhạy cảm phải dùng JWT qua header `Authorization`.
2. Backend lấy danh tính người dùng từ token, không tin vào `user_id`/`admin_id` do client gửi.
3. Chỉ admin được quản lý tài khoản và duyệt nội dung.
4. Chỉ owner hoặc admin được sửa/xóa tài nguyên cá nhân.
5. ZIP testcase được kiểm tra đường dẫn trước khi giải nén để tránh Zip Slip.
6. Ảnh upload chỉ chấp nhận JPEG, PNG, WEBP và có giới hạn dung lượng.
7. `database.initialize_database` không xóa database mặc định; chỉ xóa khi chạy với `--reset`.
8. Các biến môi trường quan trọng gồm `JWT_SECRET_KEY`, `INITIAL_ADMIN_PASSWORD`, `CORS_ALLOW_ORIGINS`, `GEMINI_API_KEY`, `VITE_API_BASE_URL`.

## 5. Kiểm tra sau khi triển khai

Các lệnh kiểm tra đề xuất:

```powershell
backend\.venv\Scripts\python.exe -m py_compile backend\auth.py backend\file_manager.py backend\main.py database\initialize_database.py
cd frontend
npm.cmd run build
```

Smoke test:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:21081/docs
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:21080/
```

Nếu backend chạy ở cổng thay thế:

```powershell
$env:VITE_API_BASE_URL="http://127.0.0.1:21083"
npm.cmd run dev -- --host 127.0.0.1 --port 21084
```

