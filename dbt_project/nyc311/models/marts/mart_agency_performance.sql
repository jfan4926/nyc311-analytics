WITH base AS (
    SELECT * FROM {{ ref('stg_complaints') }}
    WHERE resolution_hours IS NOT NULL
)

SELECT
    agency,
    agency_name,
    complaint_type,
    COUNT(*)                                    AS total_complaints,
    ROUND(AVG(resolution_hours), 1)             AS avg_resolution_hours,
    ROUND(MEDIAN(resolution_hours), 1)          AS median_resolution_hours,
    ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP
        (ORDER BY resolution_hours), 1)         AS p90_resolution_hours,
    ROUND(AVG(CASE WHEN resolution_hours > 168
        THEN 1.0 ELSE 0.0 END) * 100, 1)       AS pct_over_1_week
FROM base
GROUP BY 1, 2, 3
HAVING COUNT(*) >= 50
