# Exit Ticket

## 1. Case nào nên dùng multi-agent? Vì sao?

Khi task đòi hỏi các bước tách biệt về loại công việc *và* mỗi bước hưởng lợi từ việc được
kiểm tra độc lập trước khi đi tiếp — ví dụ research-and-write ở đây: retrieval (Researcher)
và verification (Analyst) tạo ra ghi chú có trích dẫn *trước khi* Writer viết, nên Writer khó
"bịa" claim không có nguồn. Benchmark trong repo này đo được điều đó cụ thể: multi-agent đạt
citation coverage 25-42%, baseline đạt 0% trên cùng query. Multi-agent cũng hợp lý khi task
có thể chạy bất đồng bộ/batch (không cần trả lời tức thì), vì latency cao hơn ~3x là chấp
nhận được.

## 2. Case nào không nên dùng multi-agent? Vì sao?

Khi task đơn giản, một lượt suy luận là đủ, hoặc khi latency là ràng buộc cứng (chatbot cần
trả lời real-time). Thêm agent chỉ tăng chi phí (số lượt gọi LLM tỉ lệ thuận với cost — xem
`estimated_cost_usd` trong `reports/benchmark_report.md`, multi-agent tốn gấp ~3x baseline)
và latency mà không cải thiện chất lượng, vì không có "nhu cầu xác minh khác nhau" nào giữa
các bước để multi-agent khai thác. Theo đúng `working_thesis_for_evaluation` trong corpus
offline: coordination overhead sẽ xoá sạch lợi ích chất lượng nếu task decomposition không
tạo ra thông tin/verification thực sự khác nhau.

## Failure mode quan sát được và cách fix

**Failure mode**: `ResearcherAgent` có thể trả về `state.sources = []` nếu offline corpus
không có topic nào khớp đủ từ khoá với query (query quá ngắn hoặc dùng thuật ngữ không có
trong 30 topic của corpus). Ban đầu, `SupervisorAgent` sẽ liên tục route về lại `researcher`
vì điều kiện dừng của nó (`not state.sources or not state.research_notes`) không bao giờ được
thoả — dẫn tới vòng lặp cho tới khi chạm `max_iterations`, tốn 6 lượt gọi LLM vô ích.

**Fix**: thêm guard `_repeated_failure` trong `SupervisorAgent._decide`
(`src/multi_agent_research_lab/agents/supervisor.py`) — nếu 2 route liên tiếp giống nhau
*và* `state.errors` không rỗng (nghĩa là lần chạy trước đã tự ghi nhận lỗi, ví dụ
`"researcher: no sources returned"`), Supervisor dừng ngay ở `done` thay vì thử lại thêm.
Điều này biến một vòng lặp tốn 6 lượt gọi thành dừng sau đúng 2 lượt, và lỗi vẫn được giữ lại
trong `state.errors` để benchmark tính vào `failure_rate`.
