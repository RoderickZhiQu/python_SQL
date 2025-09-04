USE movie_analytics;

-- 简化的用户统计
DROP TABLE IF EXISTS user_stats;
CREATE TABLE user_stats AS
SELECT 
    username,
    COUNT(*) as total_reviews,
    ROUND(AVG(star), 2) as avg_rating,
    SUM(like_count) as total_likes,
    'Active' as user_type
FROM movie_reviews_clean
GROUP BY username
LIMIT 1000;

-- 验证用户统计
SELECT COUNT(*) as user_count FROM user_stats;
SELECT * FROM user_stats LIMIT 5;

-- 简化的电影统计
DROP TABLE IF EXISTS movie_stats;
CREATE TABLE movie_stats AS
SELECT 
    movie_name_cn as movie_name,
    COUNT(*) as total_reviews,
    ROUND(AVG(star), 2) as avg_rating,
    SUM(like_count) as total_likes,
    ROUND(AVG(star) * 20, 2) as quality_score
FROM movie_reviews_clean
GROUP BY movie_name_cn;

-- 验证电影统计
SELECT COUNT(*) as movie_count FROM movie_stats;
SELECT * FROM movie_stats;
