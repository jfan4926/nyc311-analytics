WITH source AS (
    SELECT * FROM read_parquet('../../../data/raw/nyc311.parquet')
),

cleaned AS (
    SELECT
        unique_key,

        -- 时间字段
        created_date::TIMESTAMP AS created_at,
        closed_date::TIMESTAMP  AS closed_at,
        due_date::TIMESTAMP     AS due_at,

        -- 计算处理时长（小时）← ML的target
        CASE
            WHEN closed_date IS NOT NULL
            THEN DATEDIFF('hour', created_date::TIMESTAMP, closed_date::TIMESTAMP)
        END AS resolution_hours,

        -- 核心分类字段
        UPPER(TRIM(agency))          AS agency,
        agency_name,
        UPPER(TRIM(complaint_type))  AS complaint_type,
        descriptor,
        status,

        -- 地理字段
        CASE
            WHEN borough = 'Unspecified' THEN NULL
            ELSE UPPER(TRIM(borough))
        END AS borough,
        incident_zip,
        latitude::DOUBLE  AS latitude,
        longitude::DOUBLE AS longitude,
        community_board,

        -- 渠道
        open_data_channel_type AS channel_type,

        -- 时间特征（后面ML用）
        EXTRACT(HOUR FROM created_date::TIMESTAMP)  AS created_hour,
        EXTRACT(DOW  FROM created_date::TIMESTAMP)  AS created_dow,
        EXTRACT(MONTH FROM created_date::TIMESTAMP) AS created_month,
        CASE
            WHEN EXTRACT(DOW FROM created_date::TIMESTAMP) IN (0, 6)
            THEN TRUE ELSE FALSE
        END AS is_weekend

    FROM source
    WHERE
        unique_key IS NOT NULL
        AND created_date IS NOT NULL
        -- 过滤掉异常处理时长（负数或超过1年）
        AND (
            closed_date IS NULL
            OR DATEDIFF('hour', created_date::TIMESTAMP, closed_date::TIMESTAMP)
               BETWEEN 0 AND 8760
        )
)

SELECT * FROM cleaned
