-- 这个mart专门给ML pipeline用，一行一个complaint
WITH base AS (
    SELECT * FROM {{ ref('stg_complaints') }}
    WHERE borough IS NOT NULL
),

-- 历史平均时长（按agency+complaint_type）用来做特征
historical_avg AS (
    SELECT
        agency,
        complaint_type,
        ROUND(AVG(resolution_hours), 1) AS hist_avg_resolution_hours
    FROM base
    WHERE resolution_hours IS NOT NULL
    GROUP BY 1, 2
)

SELECT
    b.unique_key,
    b.created_at,
    b.resolution_hours,         -- ML target
    b.agency,
    b.complaint_type,
    b.borough,
    b.channel_type,
    b.created_hour,
    b.created_dow,
    b.created_month,
    b.is_weekend,
    b.incident_zip,
    COALESCE(h.hist_avg_resolution_hours, 0) AS hist_avg_resolution_hours,
    CASE
        WHEN b.resolution_hours IS NULL THEN NULL
        WHEN b.resolution_hours > 168 THEN 1  -- 超过1周 = 超时
        ELSE 0
    END AS is_overdue                         -- 分类任务target
FROM base b
LEFT JOIN historical_avg h
    ON b.agency = h.agency
    AND b.complaint_type = h.complaint_type
