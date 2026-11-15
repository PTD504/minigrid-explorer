# Task 1 Report: Channel-by-Channel Memory Blanking (`ppo_medium_5seeds_500k` on Medium)

- **Model:** `ppo_medium_5seeds_500k` (5 training seeds)
- **Test Split:** `medium`, 100 episodes (`2000000..2000099`), `deterministic=True`
- **Output Directory:** `results/eval/memory_blanking_medium/`

## 1. Bảng tổng hợp kết quả

| Cấu hình Memory | Coverage (Mean ± Std) | Δ Cov | POI (%) | Success (%) | Deaths (%) | Steps | Return | Bump (%) | Loop (%) | Eff (tiles/step) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Memory (Baseline)** | **71.83% ± 2.59%** | `0.00%` | **49.5%** | **33.6%** | 27.4% | 221.8 | **+1.573** | 1.68% | 67.44% | 1.221 |
| **Blank Ch 3 (`MEM_KNOWN_POI`)** | 70.08% ± 1.77% | `-1.75%` | 48.9% | 31.8% | 31.4% | 213.9 | +1.473 | 1.44% | 67.22% | **1.222** |
| **Blank Ch 1 (`MEM_VISITED`)** | 65.07% ± 3.22% | `-6.76%` | 43.0% | 23.2% | 34.2% | 236.0 | +0.701 | 1.23% | 71.32% | 1.039 |
| **Blank Ch 0 (`MEM_REVEALED`)** | 52.34% ± 14.72% | `-19.49%` | 32.7% | 10.4% | 42.2% | 242.2 | -0.617 | **9.26%** | 69.92% | 0.748 |
| **Blank Ch 2 (`MEM_KNOWN_WALL`)** | **35.42% ± 8.57%** | **`-36.41%`** | **17.8%** | **1.4%** | **49.6%** | 241.9 | **-2.068** | 2.51% | **83.54%** | **0.580** |
| **Blank All (`0, 1, 2, 3`)** | 61.33% ± 6.12% | `-10.50%` | 40.0% | 12.2% | **23.8%** | 310.5 | +0.302 | **0.20%** | **59.88%** | 0.558 |

---

## 2. Phân tích & Kết luận chính

1. **Thứ tự phụ thuộc vào từng kênh memory (từ cao xuống thấp):**
   `MEM_KNOWN_WALL (Ch 2)` $\gg$ `MEM_REVEALED (Ch 0)` $>$ `MEM_VISITED (Ch 1)` $\gg$ `MEM_KNOWN_POI (Ch 3)`

2. **`MEM_KNOWN_WALL` (Ch 2) là kênh xương sống:**
   * Khi blank kênh tường, Coverage rơi tự do `-36.41%` (xuống `35.42%`), Success gần như về 0 (`1.4%`), Loop Rate tăng vọt lên `83.54%`, và tỷ lệ chết tăng lên `49.6%`.
   * Điều này chứng minh `PPO+Memory` sử dụng bản đồ tường tích lũy làm khung định hướng chính để thoát phòng/hành lang.

3. **`MEM_REVEALED` (Ch 0) quyết định khả năng tìm frontier:**
   * Khi blank kênh `revealed`, Coverage giảm `-19.49%` với phương sai rất lớn giữa các seed (`± 14.72%`), và Wall Bump tăng gấp 5.5 lần (`1.68%` → `9.26%`).

4. **`MEM_VISITED` (Ch 1) hỗ trợ tinh chỉnh điều hướng:**
   * Tác động ở mức vừa (`-6.76%` coverage, `-10.4%` success), giúp giảm lặp lại đường cũ.

5. **`MEM_KNOWN_POI` (Ch 3) gần như không được sử dụng:**
   * Blank kênh POI chỉ làm lệch `-1.75%` coverage (trong biên độ nhiễu), hiệu suất khám phá giữ nguyên (`1.222` vs `1.221 tiles/step`). Nguyên nhân là reward khám phá POI được trao ngay khi nhìn thấy trong `view`, nên POI đã lưu trong memory không còn tác dụng dẫn đường.

6. **Hiện tượng "Blank All (`61.33%`) tốt hơn Blank Ch 2 (`35.42%`)":**
   * Khi xóa toàn bộ memory, nhánh `memory_cnn` chỉ xuất ra bias hằng số, policy quay về hành vi thuần `view` (đi chậm hơn: `310.5` steps, `Eff=0.558`). Ngược lại, khi *chỉ* xóa `wall` hoặc `revealed` trong khi vẫn giữ các kênh còn lại, sự mâu thuẫn không gian giữa các kênh tạo ra feature nhiễu loạn khiến agent kẹt vòng lặp (`83.54%`) hoặc đâm tường (`9.26%`).

