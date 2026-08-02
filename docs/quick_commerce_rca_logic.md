# quick_commerce_rca_logic

Business: quick_commerce
Description: Use this playbook when investigating why a store or city underperformed on a given day. Trigger when: breached_rate is high for a store-day, avg_or2a is elevated, or a user asks "why did store X perform poorly."

Start at store × day level to identify the problem, then drill into problem hours for root cause. All three checks run independently — root causes compound, they are not mutually exclusive.

Metrics: When rolling up from hour to store-day level, weight all rate/average metrics by total_orders:
- Weighted breached_rate = SUM(breached_count) / SUM(total_orders)
- Weighted avg_or2a = SUM(avg_or2a × total_orders) / SUM(total_orders)

## Data Source

Primary table: quick_commerce_orders_gold (grain: store × hour)

## Metrics

When rolling up from hour to store-day level, weight all rate/average metrics by total_orders:

- Weighted breached_rate = SUM(breached_count) / SUM(total_orders)
- Weighted avg_or2a = SUM(avg_or2a × total_orders) / SUM(total_orders)

Key columns used:

- total_orders, breached_count, breached_rate, is_problem_hour
- order_projection
- pileup_count, pileup_flag
- current_size, booked_size
- completed_count, noshow_count
- rider_hours, booked_hours, man_hour

## Content: RCA Decision Tree

### Scope

Query quick_commerce_orders_gold for the store + date. Filter hours where is_problem_hour = 1 or breached_rate is elevated. These are the hours to investigate. Run ALL of the following checks on each problem hour.

### Check 1: Demand

Compare order_projection vs total_orders for each problem hour.

- If total_orders > order_projection × 1.10 → flag DEMAND SPIKE. Actual demand exceeded forecast by >10%.

### Check 2: Pileup

Check pileup_flag and pileup_count for each problem hour.

- If pileup_flag = 1 → flag PILEUP. Orders from the previous hour spilled into this hour.
- Check if pileup is continuous for 3+ consecutive hours. If yes → flag SUSTAINED PILEUP.

### Check 3: Supply

Run both sub-checks independently.

**L1 — Booking Gap:**
Check current_capacity_booked (= booked_size / current_size) for problem hours.

- If current_capacity_booked < 0.90 → flag BOOKING GAP. Less than 90% of available slots were filled.

**L2 — Utilization Gap:**
Check man_hour (= rider_hours_per_hour / booked_hours_per_hour) for problem hours.

- If man_hour < 0.85 → flag UTILIZATION GAP. Less than 85% of booked hours were utilized on ground.
- Also check noshow_count to confirm if riders booked but didn't show up.

### Output Format

```markdown
### [Store Name] — Hour [X] — avg OR2A: [Y] min (threshold: [Z] min)

1. Demand Spike: [YES/NO] — [total_orders] orders vs [order_projection] projected ([+/- %]%)
2. Pileup: [YES/NO] — [pileup_count] orders carried from previous hour [+ SUSTAINED if 3+ consecutive hours]
3. Supply:
   a. Booking: [booked_size] of [current_size] slots booked ([X]%)
   b. Utilization: man_hour ratio [X] ([noshow_count] no-shows)

**Summary**: OR2A was [X] min over threshold due to [list of triggered flags in plain language].
```

Always answer all three checks, even when the answer is NO. The summary line must name every contributing factor in one sentence.