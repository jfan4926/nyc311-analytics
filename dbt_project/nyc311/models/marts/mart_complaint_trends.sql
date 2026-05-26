WITH base AS (
    SELECT * FROM {{ ref('stg_complaints') }}
    WHERE borough IS NOT NULL
)

SELECT
    DATE_TRUNC('week', created_at)  AS week,
    borough,
    complaint_type,
    COUNT(*)                         AS total_complaints,
    COUNT(closed_at)                 AS closed_complaints,
    ROUND(AVG(resolution_hours), 1)  AS avg_resolution_hours,
    ROUND(MEDIAN(resolution_hours), 1) AS median_resolution_hours,
    ROUND(
        COUNT(closed_at) * 100.0 / COUNT(*), 1
    )                                AS closure_rate_pct
FROM base
GROUP BY 1, 2, 3
