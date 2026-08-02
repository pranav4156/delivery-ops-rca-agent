    # order_ready_to_assignment

Business: quick_commerce
Description: Time in minutes between an order becoming ready at the store and a rider being assigned to it. Primary SLA metric for delivery operations.
Metrics: DATEDIFF(minute, shipment_ready_sent_qc, first_assigned_at) per order.
Aggregated as AVG(or2a) at store-hour level for performance analysis. All columns present in quick_commerce_order_dump

## Grain

Per order (raw), typically aggregated to store × hour.

## Source Table

- quick_commerce_order_dump: shipment_ready_sent_qc, first_assigned_at, or2a

## Thresholds

- Healthy: avg OR2A ≤ 0 (client-specific, typically 5–8 min)
- Breach: individual order where OR2A > 0
- Breach Rate threshold (BRT): if >0% of orders in a store-hour breach, flag for RCA

## Used In

- RCA Playbook (primary metric — Step 1: identify problem hours)

## Notes

- OR2A is calculated only for orders that reached assignment. Orders cancelled before assignment have no OR2A.
- Pileup inflates OR2A for the receiving hour, not the hour that caused the backlog.
- Do not average OR2A across stores without weighting by order volume.