USE movie_analytics;

-- 设置基础参数
SET hive.exec.dynamic.partition=true;
SET mapreduce.job.reduces=1;

-- 1. 基础统计
SELECT '=== 基础统计 ===' as info;
SELECT 
    COUNT(*) as total_records,
    COUNT(DISTINCT username) as unique_users,
    ROUND(AVG(star), 2) as avg_rating,
    ROUND(AVG(like_count), 1) as avg_likes,
    MIN(star) as min_rating,
    MAX(star) as max_rating
FROM movie_reviews_clean;

-- 2. 评分分布
SELECT '=== 评分分布 ===' as info;
SELECT 
    star,
    COUNT(*) as count
FROM movie_reviews_clean
GROUP BY star
ORDER BY star;

-- 3. 用户活跃度Top 10
SELECT '=== 活跃用户Top 10 ===' as info;
SELECT 
    username,
    COUNT(*) as review_count,
    ROUND(AVG(star), 2) as avg_rating,
    SUM(like_count) as total_likes
FROM movie_reviews_clean
GROUP BY username
ORDER BY review_count DESC
LIMIT 10;

-- 4. 点赞数分布统计
SELECT '=== 点赞分布统计 ===' as info;
SELECT 
    CASE 
        WHEN like_count = 0 THEN '0'
        WHEN like_count <= 100 THEN '1-100' 
        WHEN like_count <= 500 THEN '101-500'
        WHEN like_count <= 1000 THEN '501-1000'
        ELSE '1000+'
    END as like_range,
    COUNT(*) as count
FROM movie_reviews_clean
GROUP BY 
    CASE 
        WHEN like_count = 0 THEN '0'
        WHEN like_count <= 100 THEN '1-100'
        WHEN like_count <= 500 THEN '101-500' 
        WHEN like_count <= 1000 THEN '501-1000'
        ELSE '1000+'
    END;

-- 5. 评论长度分析
SELECT '=== 评论长度分析 ===' as info;
SELECT 
    CASE 
        WHEN comment_length <= 20 THEN 'Short'
        WHEN comment_length <= 100 THEN 'Medium'
        WHEN comment_length <= 200 THEN 'Long'
        ELSE 'Very Long'
    END as length_category,
    COUNT(*) as count,
    ROUND(AVG(star), 2) as avg_rating
FROM movie_reviews_clean
WHERE comment_length > 0
GROUP BY 
    CASE 
        WHEN comment_length <= 20 THEN 'Short'
        WHEN comment_length <= 100 THEN 'Medium'
        WHEN comment_length <= 200 THEN 'Long'
        ELSE 'Very Long'
    END;

SELECT '=== Hive分析完成 ===' as status;
