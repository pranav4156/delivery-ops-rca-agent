# quick_commerce_orders_gold

Business: quick_commerce
Columns:   • charge_date (date): Order date — primary date filter
  • store (varchar): Store/facility code
  • city (varchar): Order city
  • hour (float): Hour of day
  • hour_start (timestamp): Hour window start
  • hour_end (timestamp): Hour window end
  • total_orders (bigint): Total orders in store-hour
  • breached_count (bigint): Orders where OR2A exceeded SLA
  • breached_rate (numeric): breached_count / total_orders
  • is_problem_hour (int): 1 if breached_rate > threshold
  • pileup_count (bigint): Unassigned orders carried from previous hour
  • pileup_flag (int): 1 if pileup exists
  • avg_or2a (float): Average OR2A for the store-hour
  • o2a (float): Average O2A for the store-hour
  • completed_order_count (bigint): Delivered orders
  • cancelled_order_count (bigint): Cancelled orders
  • rto_order_count (bigint): RTO orders
  • order_projection (float): Forecasted order volume for the hour
  • slot_name (varchar): Slot name
  • slot_start (time): Slot start time
  • slot_end (time): Slot end time
  • slot_type (varchar): Slot type
  • orginal_size (int): Initial rider capacity set by central team
  • current_size (int): Final rider capacity set by store manager
  • booked_size (bigint): Riders who actually booked the slot
  • completed_count (bigint): Riders who showed up and completed slot
  • incompleted_count (bigint): Riders who didn't complete slot
  • noshow_count (bigint): Riders who didn't show up
  • cancelled_count (bigint): Riders whose slot was cancelled
  • orginal_capacity_booked (numeric): booked_size / orginal_size
  • current_capacity_booked (numeric): booked_size / current_size
  • rider_hours_per_hour (bigint, minutes): Actual rider hours on ground — divide by 60
  • booked_hours_per_hour (bigint, minutes): Total scheduled slot hours — divide by 60
  • man_hour (numeric): rider_hours_per_hour / booked_hours_per_hour (utilization ratio)
Data Quality Notes:   • rider_hours_per_hour and booked_hours_per_hour are in minutes despite column names — divide by 60 for hours
  • orginal_size and orginal_capacity_booked are misspelled (orginal, not original) — use as-is in queries
  • is_problem_hour is 1 when breached_rate exceeds configured threshold
  • man_hour = rider_hours_per_hour / booked_hours_per_hour (utilization ratio, not an actual hour value)
Description: Pre-aggregated gold table for hourly store performance — combines order metrics, rider capacity, slot info, and projections in one table. Primary table for RCA, performance monitoring, and capacity analysis.
Schema: hyperlocal_data_science
Use Cases:   • RCA: identify problem hours, check pileup, capacity gaps, show-up rates
  • Capacity planning: compare orginal_size → current_size → booked_size → completed_count funnel
  • Demand vs supply: order_projection vs total_orders vs booked_size

## Mandatory Filters

- Join with quick_commerce_sfm_store_mapping to apply ops filter
- Use charge_date for date filtering