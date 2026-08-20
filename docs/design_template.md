# Design Template

## Problem

Xây dựng một research assistant nhận một câu hỏi nghiên cứu dài (vd: "Research GraphRAG
state-of-the-art and write a 500-word summary"), tìm nguồn thông tin liên quan, phân tích
các claim/mâu thuẫn giữa nguồn, và viết một câu trả lời cuối cùng có trích dẫn nguồn rõ ràng
cho đối tượng "technical learners".

## Why multi-agent?

Task này có ba loại công việc khác nhau về bản chất: (1) **retrieval** — tìm và lọc nguồn
liên quan, cần khả năng dùng tool tìm kiếm; (2) **verification/synthesis** — so sánh các
claim, phát hiện mâu thuẫn, đánh giá độ tin cậy bằng chứng, cần một "góc nhìn khác" độc lập
với bước viết; (3) **writing** — trình bày lại cho đúng audience, văn phong khác hẳn hai bước
trên. Một agent duy nhất (single-agent baseline) có thể làm được cả ba, nhưng khi gộp chung
vào một prompt, model dễ bỏ qua bước so sánh bằng chứng để nhảy thẳng sang viết — baseline
benchmark của repo này cho thấy đúng vậy: baseline không trích dẫn nguồn nào
(`citation_coverage = 0%`), trong khi multi-agent đạt 25-42% nhờ Researcher/Analyst buộc
phải tạo ra ghi chú có nguồn trước khi Writer chạy. Multi-agent ở đây tách được nhu cầu xác
minh độc lập ra khỏi nhu cầu trình bày — đúng điều kiện mà multi-agent nên được dùng
(xem `working_thesis_for_evaluation` trong corpus offline).

Chi phí đánh đổi: latency multi-agent cao hơn ~3x baseline (23.4s vs 7.9s trung bình trong
benchmark chạy thử) vì có 4 lượt gọi LLM tuần tự thay vì 1. Đây là đánh đổi hợp lý cho một
tác vụ nghiên cứu chạy offline/batch, nhưng sẽ không hợp lý cho một chatbot cần trả lời tức thì.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Quyết định worker nào chạy tiếp theo và khi nào dừng | `ResearchState` (notes/answer đã có gì) | `route_history` entry mới | Route lặp vô hạn nếu state không tiến triển → guard: dừng khi 2 route liên tiếp giống nhau kèm lỗi, hoặc khi vượt `max_iterations` |
| Researcher | Tìm nguồn (offline corpus retrieval) và viết research notes có trích dẫn | `request.query` | `state.sources`, `state.research_notes` | Không tìm được nguồn → notes rỗng, `state.errors` được ghi, Supervisor sẽ dừng sau 2 lần thử thay vì lặp mãi |
| Analyst | Trích xuất claim chính, so sánh quan điểm, gắn cờ bằng chứng yếu | `state.research_notes` | `state.analysis_notes` | Thiếu `research_notes` → trả về thông báo lỗi thay vì gọi LLM với input rỗng |
| Writer | Tổng hợp notes + analysis thành câu trả lời cuối, giữ trích dẫn | `state.research_notes`, `state.analysis_notes`, `state.sources` | `state.final_answer` | LLM lỗi/timeout → `AgentExecutionError` được bắt ở node graph, ghi vào `state.errors`, Supervisor route sang `done` thay vì crash toàn bộ workflow |
| Critic (bonus) | Kiểm tra citation coverage của final answer, phát hiện trích dẫn không tồn tại trong `state.sources` | `state.final_answer`, `state.sources` | `AgentResult` với `citation_coverage`, cảnh báo trong `state.errors` nếu có unknown citation | Không có failure mode nghiêm trọng — chạy sau cùng, chỉ đọc state |

## Shared state

`ResearchState` (xem `src/multi_agent_research_lab/core/state.py`):

- `request`: câu hỏi gốc + audience + max_sources — cố định trong suốt workflow.
- `iteration` / `route_history`: đếm số bước supervisor đã ra quyết định, dùng để enforce
  `max_iterations` và để debug đường đi qua các agent.
- `sources`: danh sách `SourceDocument` để Writer/Critic biết id nào hợp lệ khi trích dẫn.
- `research_notes`, `analysis_notes`, `final_answer`: presence/absence của các field này
  chính là điều kiện routing của Supervisor — không cần thêm state machine riêng.
- `agent_results`: lịch sử output + metadata (cost, tokens) của từng agent, dùng cho benchmark.
- `trace`: span có tên + attributes + duration cho từng bước, dùng để debug và tính latency.
- `errors`: danh sách lỗi tích lũy — vừa là guardrail signal (Supervisor đọc để tránh loop),
  vừa là input cho benchmark's `failure_rate`.

## Routing policy

```text
supervisor
   |
   |-- sources/research_notes rỗng  --> researcher --\
   |-- analysis_notes rỗng          --> analyst    ---+--> (quay lại supervisor)
   |-- final_answer rỗng            --> writer     ---+
   |-- critic chưa chạy             --> critic     --/
   |-- else                         --> done (END)
```

Guard bổ sung được kiểm tra trước các nhánh trên: nếu `iteration >= max_iterations`, hoặc
2 route liên tiếp giống nhau kèm `state.errors` không rỗng (nghĩa là agent đó đã fail và
không đẩy được state tiến lên), Supervisor trả về `done` ngay lập tức.

## Guardrails

- Max iterations: `Settings.max_iterations` (mặc định 6, đọc từ `MAX_ITERATIONS` trong `.env`).
- Timeout: `Settings.timeout_seconds` truyền vào `OpenAI(timeout=...)` cho mỗi call LLM.
- Retry: `LLMClient._call` dùng `tenacity` — 3 lần, exponential backoff, chỉ retry lỗi
  transient (`APITimeoutError`, `RateLimitError`, `APIError`).
- Fallback: nếu một worker raise `AgentExecutionError`, node wrapper trong
  `graph/workflow.py::_make_node` bắt lỗi, ghi vào `state.errors`, và trả state nguyên trạng
  về cho Supervisor thay vì crash toàn graph — Supervisor sau đó route sang `done` thay vì
  thử lại vô hạn.
- Validation: input qua `ResearchQuery` (Pydantic) ở CLI layer; Critic kiểm tra citation
  coverage và unknown citation ids sau khi Writer chạy xong.

## Benchmark plan

Chạy `malab benchmark` (hoặc `python -m multi_agent_research_lab.cli benchmark`) trên 3 query
mặc định trong `configs/lab_default.yaml`:

| Query | Metric kỳ vọng | Expected outcome |
|---|---|---|
| "Research GraphRAG state-of-the-art and write a 500-word summary" | citation_coverage, quality | multi-agent > baseline trên cả hai |
| "Compare single-agent and multi-agent workflows for customer support" | latency | multi-agent chậm hơn rõ rệt (nhiều LLM call hơn) |
| "Summarize production guardrails for LLM agents" | failure_rate | cả hai = 0% trong điều kiện bình thường (không rate-limit) |

Kết quả thực đo (xem `reports/benchmark_report.md`, chạy ngày 2026-08-20): multi-agent có
citation coverage trung bình 25-42% (baseline 0%), quality proxy trung bình 7.3/10 so với
6.0/10 của baseline, đổi lại latency trung bình ~23.5s so với ~7.9s.
