USE movie_analytics;

-- 设置强制执行参数
SET hive.exec.dynamic.partition=true;
SET hive.exec.dynamic.partition.mode=nonstrict;
SET mapreduce.job.reduces=1;

-- 强制清洗数据
INSERT OVERWRITE TABLE movie_reviews_clean
SELECT 
    COALESCE(id, CAST(RAND() * 1000000 AS STRING)) as id,
    COALESCE(movie_name_en, 'Unknown') as movie_name_en,
    COALESCE(movie_name_cn, 'Unknown') as movie_name_cn,
    COALESCE(crawl_date, '2017-01-22') as crawl_date,
    COALESCE(number, 0) as number,
    COALESCE(username, 'Anonymous') as username,
    COALESCE(review_date, '2015-01-01') as review_date,
    COALESCE(star, 3.0) as star,
    COALESCE(comment, 'No comment') as comment,
    COALESCE(like_count, 0) as like_count,
    LENGTH(COALESCE(comment, 'No comment')) as comment_length
FROM movie_reviews_raw;

-- 验证清洗结果
SELECT COUNT(*) as cleaned_records FROM movie_reviews_clean;
