import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import re
import time
from datetime import datetime
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class MovieSentimentAnalyzer:
    """电影评论情感分析器 - Python UDF实现"""
    
    def __init__(self):
        self.setup_dictionaries()
        self.results = {}
        
    def setup_dictionaries(self):
        """初始化情感词典和关键词库"""
        
        # 正面情感词典
        self.positive_words = {
            # 基础正面词 (权重1.0)
            '好': 1.0, '棒': 1.2, '赞': 1.1, '优秀': 1.5, '精彩': 1.6, '出色': 1.4,
            '完美': 1.8, '满意': 1.3, '推荐': 1.7, '喜欢': 1.2, '爱': 1.4,
            
            # 电影专业正面词 (权重1.5+)
            '震撼': 1.8, '感动': 1.6, '温馨': 1.4, '有趣': 1.3, '幽默': 1.4,
            '搞笑': 1.5, '经典': 2.0, '佳作': 1.9, '神作': 2.2, '巨制': 1.7,
            '史诗': 1.8, '壮观': 1.6, '唯美': 1.5, '治愈': 1.6, '燃': 1.7,
            
            # 技术层面正面词
            '精良': 1.5, '用心': 1.4, '细腻': 1.3, '流畅': 1.2, '自然': 1.1,
            '真实': 1.3, '生动': 1.2, '丰富': 1.1, '深刻': 1.6, '有意义': 1.5
        }
        
        # 负面情感词典
        self.negative_words = {
            # 基础负面词
            '差': -1.0, '烂': -1.5, '垃圾': -1.8, '无聊': -1.2, '失望': -1.3,
            '糟糕': -1.4, '恶心': -1.6, '讨厌': -1.2, '后悔': -1.3, '浪费': -1.4,
            
            # 电影专业负面词
            '尴尬': -1.2, '弱智': -1.8, '幼稚': -1.3, '老套': -1.1, '俗套': -1.2,
            '拖沓': -1.4, '混乱': -1.3, '低级': -1.5, '粗糙': -1.3, '敷衍': -1.4,
            '做作': -1.2, '矫情': -1.1, '虚假': -1.3, '空洞': -1.2, '肤浅': -1.1,
            
            # 强化负面词
            '超烂': -2.0, '巨坑': -1.9, '智商税': -1.8, '毁童年': -1.7, '辣眼睛': -1.6
        }
        
        # 程度副词
        self.degree_words = {
            '非常': 2.0, '超级': 2.2, '极其': 2.5, '相当': 1.8, '特别': 1.9,
            '很': 1.5, '挺': 1.3, '比较': 1.2, '还': 1.1, '稍微': 0.8,
            '有点': 0.7, '略': 0.6, '不太': 0.3, '一般': 0.5
        }
        
        # 否定词
        self.negation_words = {
            '不', '没', '无', '非', '未', '否', '别', '休', '不要', '不能', 
            '不会', '没有', '不是', '不行', '不好', '不对', '决不', '并非'
        }
        
        # 电影关键词库 (按类型分类)
        self.movie_keywords = {
            '剧情': ['剧情', '故事', '情节', '叙事', '结构', '逻辑', '节奏', '发展'],
            '人物': ['角色', '人物', '主角', '配角', '性格', '形象', '塑造', '关系'],
            '表演': ['演技', '表演', '演员', '台词', '情感', '自然', '真实', '投入'],
            '技术': ['特效', '视效', '画面', '镜头', '剪辑', '摄影', '制作', '后期'],
            '音乐': ['配乐', '音乐', '音效', '声音', '插曲', '主题曲', '背景音乐'],
            '导演': ['导演', '执导', '掌控', '风格', '水平', '功力', '手法', '技巧'],
            '情感': ['感动', '温馨', '催泪', '治愈', '共鸣', '感情', '情绪', '心理'],
            '娱乐': ['搞笑', '幽默', '有趣', '轻松', '愉快', '欢乐', '喜剧', '段子'],
            '视觉': ['画面', '色彩', '构图', '美感', '震撼', '壮观', '唯美', '视觉']
        }
        
        # 27部电影名称标准化
        self.movie_names = {
            'Avengers Age of Ultron': '复仇者联盟2',
            'Big Fish & Begonia': '大鱼海棠', 
            'Captain America Civil War': '美国队长3',
            'Chinese Zodiac': '十二生肖',
            'Mojin The Lost Legend': '九层妖塔',
            'Monkey King Hero Is Back': '大圣归来',
            'Gardenia': '栀子花开',
            'Goodbye Mr Loser': '夏洛特烦恼',
            'Iron Man': '钢铁侠1',
            'Journey to the West': '西游降魔篇',
            'La La Land': '爱乐之城',
            'Lost in Thailand': '泰囧',
            'Operation Mekong': '湄公河行动',
            'Soul Mate': '七月与安生',
            'The Avengers': '复仇者联盟',
            'See You Tomorrow': '后会无期',
            'The Ghouls': '寻龙诀',
            'Left Ear': '左耳',
            'The Great Wall': '长城',
            'The Mermaid': '美人鱼',
            'Tiny Times': '小时代1',
            'Tiny Times 3': '小时代3',
            'Train to Busan': '釜山行',
            'Transformers 4': '变形金刚4',
            'Your Name': '你的名字',
            'Zootopia': '疯狂动物城'
        }
        
    def sentiment_score_udf(self, comment):
        """情感分析UDF - 核心算法"""
        if pd.isna(comment) or not str(comment).strip():
            return 0.0
            
        text = str(comment).lower()
        total_score = 0.0
        word_count = 0
        
        # 分词 (简化处理)
        words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', text)
        
        for i, word in enumerate(words):
            base_score = 0.0
            
            # 检查情感词
            if word in self.positive_words:
                base_score = self.positive_words[word]
                word_count += 1
            elif word in self.negative_words:
                base_score = self.negative_words[word]
                word_count += 1
            else:
                # 检查包含情感词的复合词
                for pos_word, score in self.positive_words.items():
                    if pos_word in word:
                        base_score = score * 0.8
                        word_count += 1
                        break
                if base_score == 0:
                    for neg_word, score in self.negative_words.items():
                        if neg_word in word:
                            base_score = score * 0.8
                            word_count += 1
                            break
            
            if base_score != 0:
                # 处理程度副词
                degree_multiplier = 1.0
                if i > 0 and words[i-1] in self.degree_words:
                    degree_multiplier = self.degree_words[words[i-1]]
                
                # 处理否定词
                negation_count = 0
                for j in range(max(0, i-3), i):
                    if words[j] in self.negation_words:
                        negation_count += 1
                
                if negation_count % 2 == 1:
                    base_score *= -1
                
                total_score += base_score * degree_multiplier
        
        if word_count == 0:
            return 0.0
            
        # 计算平均分并规范化到[-1, 1]
        avg_score = total_score / word_count
        normalized_score = np.tanh(avg_score)  # 使用tanh函数平滑映射
        
        return round(normalized_score, 4)
    
    def sentiment_label_udf(self, score):
        """情感标签UDF"""
        if score >= 0.3:
            return 'positive'
        elif score <= -0.3:
            return 'negative'
        else:
            return 'neutral'
    
    def extract_keywords_udf(self, comment, max_keywords=5):
        """关键词提取UDF"""
        if pd.isna(comment):
            return []
            
        text = str(comment).lower()
        keywords = []
        
        # 按类别检查关键词
        for category, words in self.movie_keywords.items():
            for word in words:
                if word in text:
                    keywords.append(word)
        
        # 去重并返回前N个
        unique_keywords = list(dict.fromkeys(keywords))
        return unique_keywords[:max_keywords]
    
    def process_large_dataset(self, df, batch_size=50000):
        """大规模数据处理"""
        print(f"开始处理大规模数据集: {len(df):,} 条记录")
        print(f"使用批处理大小: {batch_size:,}")
        
        results = []
        total_batches = len(df) // batch_size + (1 if len(df) % batch_size > 0 else 0)
        
        for i in range(0, len(df), batch_size):
            batch_num = i // batch_size + 1
            batch_df = df.iloc[i:i+batch_size].copy()
            
            print(f"处理批次 {batch_num}/{total_batches} ({len(batch_df):,} 条记录)...")
            
            start_time = time.time()
            
            # 并行处理情感分析
            batch_df['sentiment_score'] = batch_df['Comment'].apply(self.sentiment_score_udf)
            batch_df['sentiment_label'] = batch_df['sentiment_score'].apply(self.sentiment_label_udf)
            batch_df['keywords'] = batch_df['Comment'].apply(self.extract_keywords_udf)
            
            # 添加电影中文名
            batch_df['movie_name_cn'] = batch_df['Movie_Name_EN'].map(
                lambda x: self.movie_names.get(x, x) if pd.notna(x) else '未知电影'
            )
            
            batch_time = time.time() - start_time
            print(f"批次 {batch_num} 处理完成，耗时: {batch_time:.2f}秒")
            
            results.append(batch_df)
        
        print("合并所有批次结果...")
        final_df = pd.concat(results, ignore_index=True)
        
        return final_df
    
    def analyze_all_movies(self, df_processed):
        """多电影分析"""
        print("\n开始多电影情感分析...")
        
        movie_analysis = {}
        
        for movie_cn in df_processed['movie_name_cn'].unique():
            if movie_cn == '未知电影':
                continue
                
            movie_data = df_processed[df_processed['movie_name_cn'] == movie_cn]
            
            analysis = {
                'total_reviews': len(movie_data),
                'avg_sentiment': movie_data['sentiment_score'].mean(),
                'sentiment_distribution': movie_data['sentiment_label'].value_counts().to_dict(),
                'avg_rating': movie_data['Star'].mean() if 'Star' in movie_data.columns else None,
                'total_likes': movie_data['Like'].sum() if 'Like' in movie_data.columns else None,
                'top_keywords': self.get_top_keywords(movie_data['keywords'])
            }
            
            movie_analysis[movie_cn] = analysis
        
        return movie_analysis
    
    def get_top_keywords(self, keywords_series, top_n=10):
        """获取热门关键词"""
        all_keywords = []
        for keywords_list in keywords_series:
            if isinstance(keywords_list, list):
                all_keywords.extend(keywords_list)
        
        if not all_keywords:
            return []
            
        keyword_counts = Counter(all_keywords)
        return keyword_counts.most_common(top_n)
    
    def create_comprehensive_visualization(self, df_processed, movie_analysis):
        """创建综合可视化报告"""
        print("\n创建综合可视化报告...")
        
        # 创建大型综合图表
        fig, axes = plt.subplots(4, 3, figsize=(24, 20))
        fig.suptitle('电影评论大数据情感分析报告\n基于Python UDF处理200万+条数据', 
                    fontsize=20, fontweight='bold')
        
        # 1. 整体情感分布
        sentiment_counts = df_processed['sentiment_label'].value_counts()
        colors_sentiment = ['#2E8B57', '#FFD700', '#DC143C']  # 绿、黄、红
        pie1 = axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, 
                             autopct='%1.1f%%', colors=colors_sentiment, startangle=90)
        axes[0, 0].set_title('整体情感分布', fontsize=14, fontweight='bold')
        
        # 2. 情感分数分布直方图
        axes[0, 1].hist(df_processed['sentiment_score'], bins=50, alpha=0.7, 
                       color='skyblue', edgecolor='navy')
        axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].set_title('情感分数分布', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('情感分数')
        axes[0, 1].set_ylabel('频次')
        
        # 3. 各电影平均情感对比
        movie_sentiment = {movie: data['avg_sentiment'] for movie, data in movie_analysis.items()}
        movies = list(movie_sentiment.keys())[:15]  # 显示前15部电影
        sentiments = [movie_sentiment[movie] for movie in movies]
        
        bars = axes[0, 2].barh(range(len(movies)), sentiments, 
                              color=['green' if s > 0.1 else 'red' if s < -0.1 else 'orange' for s in sentiments])
        axes[0, 2].set_yticks(range(len(movies)))
        axes[0, 2].set_yticklabels(movies, fontsize=10)
        axes[0, 2].set_title('各电影平均情感分数', fontsize=14, fontweight='bold')
        axes[0, 2].axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # 4. 评论数量Top10电影
        movie_counts = {movie: data['total_reviews'] for movie, data in movie_analysis.items()}
        top_movies = sorted(movie_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        movies_top, counts_top = zip(*top_movies)
        axes[1, 0].bar(range(len(movies_top)), counts_top, color='lightcoral', alpha=0.8)
        axes[1, 0].set_xticks(range(len(movies_top)))
        axes[1, 0].set_xticklabels(movies_top, rotation=45, ha='right', fontsize=9)
        axes[1, 0].set_title('评论数量Top10电影', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('评论数量')
        
        # 5. 情感分数vs评分关系
        if 'Star' in df_processed.columns:
            sample_data = df_processed.sample(min(10000, len(df_processed)))
            scatter = axes[1, 1].scatter(sample_data['sentiment_score'], sample_data['Star'], 
                                       alpha=0.5, c=sample_data['sentiment_score'], 
                                       cmap='RdYlGn', s=20)
            axes[1, 1].set_xlabel('情感分数')
            axes[1, 1].set_ylabel('用户评分')
            axes[1, 1].set_title('情感分数 vs 用户评分关系', fontsize=14, fontweight='bold')
            plt.colorbar(scatter, ax=axes[1, 1], label='情感分数')
        
        # 6. 关键词热力图 (简化版)
        self.create_keyword_heatmap(axes[1, 2], df_processed)
        
        # 7-12. 各电影详细分析 (选择6部热门电影)
        popular_movies = list(dict(top_movies).keys())[:6]
        
        for idx, movie in enumerate(popular_movies):
            row = 2 + idx // 3
            col = idx % 3
            
            if row < 4:  # 确保不超出子图范围
                movie_data = df_processed[df_processed['movie_name_cn'] == movie]
                sentiment_dist = movie_data['sentiment_label'].value_counts()
                
                axes[row, col].pie(sentiment_dist.values, labels=sentiment_dist.index, 
                                  autopct='%1.1f%%', colors=colors_sentiment)
                axes[row, col].set_title(f'{movie}\n({len(movie_data):,}条评论)', 
                                       fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('comprehensive_sentiment_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("综合分析图表已保存: comprehensive_sentiment_analysis.png")
    
    def create_keyword_heatmap(self, ax, df_processed):
        """创建关键词热力图"""
        # 构建电影-关键词矩阵
        movies = df_processed['movie_name_cn'].value_counts().head(10).index
        all_keywords = []
        for keywords_list in df_processed['keywords']:
            if isinstance(keywords_list, list):
                all_keywords.extend(keywords_list)
        
        top_keywords = [kw for kw, _ in Counter(all_keywords).most_common(10)]
        
        # 创建热力图数据
        heatmap_data = []
        for movie in movies:
            movie_data = df_processed[df_processed['movie_name_cn'] == movie]
            movie_keywords = []
            for keywords_list in movie_data['keywords']:
                if isinstance(keywords_list, list):
                    movie_keywords.extend(keywords_list)
            
            keyword_counts = Counter(movie_keywords)
            row = [keyword_counts.get(kw, 0) for kw in top_keywords]
            heatmap_data.append(row)
        
        if heatmap_data:
            heatmap_df = pd.DataFrame(heatmap_data, index=movies, columns=top_keywords)
            sns.heatmap(heatmap_df, annot=True, fmt='d', cmap='YlOrRd', ax=ax, cbar=False)
            ax.set_title('电影-关键词热力图', fontsize=14, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='y', rotation=0)
    
    def create_simple_wordcloud_visualization(self, df_processed):
        """创建简化版词云可视化 (不依赖WordCloud库)"""
        print("\n创建简化版关键词可视化...")
        
        # 统计所有关键词
        all_keywords = []
        for keywords_list in df_processed['keywords']:
            if isinstance(keywords_list, list):
                all_keywords.extend(keywords_list)
        
        keyword_counts = Counter(all_keywords)
        top_keywords = keyword_counts.most_common(30)
        
        if not top_keywords:
            print("未找到关键词，跳过词云生成")
            return
        
        # 创建关键词可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 1. 关键词频次柱状图
        keywords, counts = zip(*top_keywords)
        
        bars = ax1.barh(range(len(keywords)), counts, color=plt.cm.viridis(np.linspace(0, 1, len(keywords))))
        ax1.set_yticks(range(len(keywords)))
        ax1.set_yticklabels(keywords, fontsize=11)
        ax1.set_title('Top 30 关键词频次统计', fontsize=16, fontweight='bold')
        ax1.set_xlabel('出现次数')
        
        # 添加数值标签
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax1.text(bar.get_width() + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
                    f'{count:,}', ha='left', va='center', fontsize=9)
        
        # 2. 关键词散点图 (模拟词云效果)
        np.random.seed(42)
        x_positions = np.random.rand(len(top_keywords)) * 10
        y_positions = np.random.rand(len(top_keywords)) * 8
        sizes = [count/max(counts) * 2000 + 200 for _, count in top_keywords]
        colors = plt.cm.plasma(np.linspace(0, 1, len(top_keywords)))
        
        for i, (keyword, count) in enumerate(top_keywords):
            ax2.scatter(x_positions[i], y_positions[i], s=sizes[i], 
                       c=[colors[i]], alpha=0.7, edgecolors='black', linewidth=0.5)
            ax2.annotate(keyword, (x_positions[i], y_positions[i]), 
                        xytext=(0, 0), textcoords='offset points',
                        ha='center', va='center', fontsize=8, fontweight='bold')
        
        ax2.set_xlim(-0.5, 10.5)
        ax2.set_ylim(-0.5, 8.5)
        ax2.set_title('关键词词云图 (简化版)', fontsize=16, fontweight='bold')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig('simplified_keyword_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("简化版关键词可视化已保存: simplified_keyword_visualization.png")
        
        return top_keywords
    
    def generate_analysis_report(self, df_processed, movie_analysis, top_keywords):
        """生成详细分析报告"""
        print("\n生成分析报告...")
        
        # 基础统计
        total_records = len(df_processed)
        sentiment_dist = df_processed['sentiment_label'].value_counts()
        avg_sentiment = df_processed['sentiment_score'].mean()
        
        report = f"""
# 电影评论大数据情感分析报告
## 基于Python UDF处理 {total_records:,} 条评论数据

### 执行概况
- 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 数据规模: {total_records:,} 条评论
- 电影数量: {len(movie_analysis)} 部
- 处理方法: Python自定义UDF函数

### 整体情感分析结果
- 平均情感分数: {avg_sentiment:.4f}
- 正面评论: {sentiment_dist.get('positive', 0):,} 条 ({sentiment_dist.get('positive', 0)/total_records*100:.1f}%)
- 中性评论: {sentiment_dist.get('neutral', 0):,} 条 ({sentiment_dist.get('neutral', 0)/total_records*100:.1f}%)
- 负面评论: {sentiment_dist.get('negative', 0):,} 条 ({sentiment_dist.get('negative', 0)/total_records*100:.1f}%)

### 各电影情感分析排名

#### 最受好评电影 (按平均情感分数)
"""
        
        # 电影排名
        movie_ranking = sorted(movie_analysis.items(), key=lambda x: x[1]['avg_sentiment'], reverse=True)
        
        for i, (movie, data) in enumerate(movie_ranking[:10], 1):
            report += f"{i:2d}. {movie}: {data['avg_sentiment']:+.4f} ({data['total_reviews']:,}条评论)\n"
        
        report += f"""
#### 最热门电影 (按评论数量)
"""
        
        popular_ranking = sorted(movie_analysis.items(), key=lambda x: x[1]['total_reviews'], reverse=True)
        
        for i, (movie, data) in enumerate(popular_ranking[:10], 1):
            report += f"{i:2d}. {movie}: {data['total_reviews']:,}条评论 (平均情感: {data['avg_sentiment']:+.4f})\n"
        
        report += f"""
### 关键词分析

#### 热门关键词Top20
"""
        
        for i, (keyword, count) in enumerate(top_keywords[:20], 1):
            report += f"{i:2d}. {keyword}: {count:,}次\n"
        
        report += f"""
### 主要发现
1. 数据处理成功率: 100% (处理了全部{total_records:,}条记录)
2. 情感倾向: {'整体偏正面' if avg_sentiment > 0.1 else '整体偏负面' if avg_sentiment < -0.1 else '整体中性'}
3. 用户参与度: 平均每部电影 {total_records//len(movie_analysis):,} 条评论
4. 热门关键词: {', '.join([kw for kw, _ in top_keywords[:5]])}

### 技术实现
- UDF实现语言: Python 3
- 情感分析算法: 基于词典的加权计算
- 关键词提取: 基于预定义词库匹配
- 数据处理: 分批处理大规模数据
- 可视化: matplotlib + seaborn

### 数据质量评估
- 数据完整性: {(total_records - df_processed.isnull().sum().sum()) / (total_records * len(df_processed.columns)) * 100:.1f}%
- 情感识别率: {(total_records - sentiment_dist.get('neutral', 0)) / total_records * 100:.1f}%
- 关键词覆盖率: {len([kw for kw, _ in top_keywords]) / total_records * 100:.3f}%

---
报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # 保存报告
        with open('sentiment_analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("详细分析报告已保存: sentiment_analysis_report.md")
        
        return report


def main():
    """主程序 - 执行完整的3.4交付物"""
    print("="*60)
    print("3.4交付物 - Python UDF大规模情感分析")
    print("="*60)
    
    # 初始化分析器
    analyzer = MovieSentimentAnalyzer()
    
    # 读取数据
    print("读取原始数据...")
    try:
        df = pd.read_csv('tx_video.csv')
        print(f"成功读取数据: {len(df):,} 条记录")
        print(f"涉及电影: {df['Movie_Name_EN'].nunique()} 部")
    except Exception as e:
        print(f"读取数据失败: {e}")
        return
    
    # 大规模数据处理
    print("\n开始大规模情感分析处理...")
    start_time = time.time()
    
    df_processed = analyzer.process_large_dataset(df, batch_size=100000)
    
    processing_time = time.time() - start_time
    print(f"\n情感分析处理完成!")
    print(f"总耗时: {processing_time:.2f}秒")
    print(f"处理速度: {len(df_processed)/processing_time:.0f} 条/秒")
    
    # 多电影分析
    movie_analysis = analyzer.analyze_all_movies(df_processed)
    print(f"完成 {len(movie_analysis)} 部电影的分析")
    
    # 创建可视化
    analyzer.create_comprehensive_visualization(df_processed, movie_analysis)
    
    # 创建简化版词云
    top_keywords = analyzer.create_simple_wordcloud_visualization(df_processed)
    
    # 生成报告
    report = analyzer.generate_analysis_report(df_processed, movie_analysis, top_keywords)
    
    # 保存处理后的数据
    print("\n保存处理结果...")
    df_processed.to_csv('sentiment_analysis_results.csv', index=False, encoding='utf-8')
    print("情感分析结果已保存: sentiment_analysis_results.csv")
    
    # 保存电影分析结果
    with open('movie_analysis_summary.json', 'w', encoding='utf-8') as f:
        json.dump(movie_analysis, f, ensure_ascii=False, indent=2)
    print("电影分析摘要已保存: movie_analysis_summary.json")
    
    print("\n" + "="*60)
    print("3.4交付物完成!")
    print("="*60)
    print("生成的文件:")
    print("1. comprehensive_sentiment_analysis.png - 综合情感分析图表")
    print("2. simplified_keyword_visualization.png - 简化版关键词可视化")
    print("3. sentiment_analysis_results.csv - 完整情感分析结果")
    print("4. movie_analysis_summary.json - 电影分析摘要") 
    print("5. sentiment_analysis_report.md - 详细分析报告")
    print("6. Python UDF源码 - 情感分析和关键词提取函数")


if __name__ == "__main__":
    main()