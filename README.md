# AI-CLASS-FINAL-PROJECT-LUOXIAOYI-CODE-EXAMPLE

import nltk
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import matplotlib.pyplot as plt
import seaborn as sns

# 下载必要的NLTK数据
nltk.download('punkt')

# 改进的情感分析函数，支持批量处理
def text_emotion_large_text(text, chunk_size=50):
    '''
    INPUT: 长文本字符串, chunk_size(每多少句子作为一个分析单元)
    OUTPUT: 包含情感分析结果的DataFrame
    '''
    
    # 读取情感词典
    csv_path = r'C:\Program Files\NRC-Emotion-Lexicon-v0.92-In105Languages-Nov2017Translations.csv'
    try:
        emolex_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"错误：找不到情感词典文件 {csv_path}")
        print("请确保NRC情感词典文件路径正确")
        return None
    
    # 使用正确的列名
    emolex_df = emolex_df[['English (en)', 'Positive','Negative','Anger', 'Anticipation', 'Disgust', 'Fear','Joy',
                      'Sadness', 'Surprise', 'Trust']]
    emolex_df = emolex_df.rename(columns={'English (en)': 'English'})
    
    emotions = emolex_df.columns.drop('English')
    stemmer = SnowballStemmer("english")
    
    # 文本预处理函数
    def preprocess_text(text):
        # 分割成句子
        sentences = sent_tokenize(text)
        
        # 清理文本：移除多余的空白字符和特殊字符
        cleaned_sentences = []
        for sentence in sentences:
            # 移除数字和特殊字符，只保留字母和基本标点
            cleaned = re.sub(r'[^a-zA-Z\s\.\!\?]', ' ', sentence)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            if len(cleaned) > 10:  # 只保留有意义的句子
                cleaned_sentences.append(cleaned)
        
        return cleaned_sentences
    
    # 分析单个文本块的情感
    def analyze_chunk(chunk_text, chunk_id):
        emotions_score = {emotion: 0 for emotion in emotions}
        emotions_score['chunk_id'] = chunk_id
        emotions_score['text'] = chunk_text
        emotions_score['word_count'] = len(word_tokenize(chunk_text))
        
        words = word_tokenize(chunk_text.lower())
        for word in words:
            word_stem = stemmer.stem(word)
            emo_score = emolex_df[emolex_df.English == word_stem]
            if not emo_score.empty:
                for emotion in emotions:
                    emotions_score[emotion] += emo_score[emotion].iloc[0]
        
        return emotions_score
    
    # 主处理流程
    print("正在进行文本预处理...")
    sentences = preprocess_text(text)
    print(f"预处理完成，共 {len(sentences)} 个句子")
    
    # 将句子分组为块
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = ' '.join(sentences[i:i+chunk_size])
        chunks.append(chunk)
    
    print(f"将文本分为 {len(chunks)} 个分析块")
    
    # 分析每个块
    results = []
    print("开始情感分析...")
    for i, chunk in enumerate(tqdm(chunks)):
        if chunk.strip():  # 确保块不为空
            result = analyze_chunk(chunk, i)
            results.append(result)
    
    # 创建结果DataFrame
    result_df = pd.DataFrame(results)
    
    # 重新排列列的顺序
    columns = ['chunk_id', 'text', 'word_count'] + list(emotions)
    result_df = result_df[columns]
    
    return result_df

# 读取《老人与海》文本文件
def read_old_man_and_sea(file_path):
    """
    读取小说文本文件
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        print(f"成功读取文件，文本长度: {len(text)} 字符")
        return text
    except UnicodeDecodeError:
        print("UTF-8编码失败，尝试使用latin-1编码...")
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                text = file.read()
            print(f"使用latin-1编码成功读取文件，文本长度: {len(text)} 字符")
            return text
        except Exception as e:
            print(f"读取文件失败: {e}")
            return None
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None

# 分析整个小说的情感趋势
def analyze_novel_emotion_trends(result_df):
    """
    分析小说的情感趋势
    """
    emotion_columns = ['Positive', 'Negative', 'Anger', 'Anticipation', 'Disgust', 
                      'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust']
    
    # 计算总体情感统计
    total_stats = {}
    for emotion in emotion_columns:
        total_stats[f'total_{emotion.lower()}'] = result_df[emotion].sum()
        total_stats[f'avg_{emotion.lower()}_per_chunk'] = result_df[emotion].mean()
        total_stats[f'max_{emotion.lower()}'] = result_df[emotion].max()
    
    # 找到情感高峰的段落
    peak_emotions = {}
    for emotion in emotion_columns:
        idx = result_df[emotion].idxmax()
        peak_emotions[emotion] = {
            'chunk_id': result_df.loc[idx, 'chunk_id'],
            'value': result_df.loc[idx, emotion],
            'text_preview': result_df.loc[idx, 'text'][:100] + '...'
        }
    
    return total_stats, peak_emotions

# 可视化情感趋势
def visualize_emotion_trends(result_df, novel_title="The Old Man and the Sea"):
    """
    可视化情感趋势
    """
    # 设置图表样式
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'《{novel_title}》情感分析结果', fontsize=16, fontweight='bold')
    
    # 1. 主要情感趋势
    emotions_plot = ['Positive', 'Negative', 'Joy', 'Sadness', 'Fear']
    colors = ['green', 'red', 'orange', 'blue', 'purple']
    for i, emotion in enumerate(emotions_plot):
        axes[0, 0].plot(result_df['chunk_id'], result_df[emotion], 
                       label=emotion, linewidth=2, color=colors[i], alpha=0.8)
    axes[0, 0].set_title('主要情感趋势 throughout the Novel')
    axes[0, 0].set_xlabel('文本块编号 (故事进展)')
    axes[0, 0].set_ylabel('情感强度')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 情感分布饼图
    emotion_totals = result_df[emotions_plot].sum()
    wedges, texts, autotexts = axes[0, 1].pie(emotion_totals, labels=emotion_totals.index, 
                                            autopct='%1.1f%%', startangle=90, colors=colors)
    axes[0, 1].set_title('情感分布比例')
    
    # 3. 所有情感的箱线图
    emotion_columns = ['Positive', 'Negative', 'Anger', 'Anticipation', 'Disgust', 
                      'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust']
    emotion_data = result_df[emotion_columns]
    box_plot = axes[1, 0].boxplot([emotion_data[col] for col in emotion_columns], 
                                labels=emotion_columns, patch_artist=True)
    
    # 为箱线图添加颜色
    colors_box = plt.cm.Set3(range(len(emotion_columns)))
    for patch, color in zip(box_plot['boxes'], colors_box):
        patch.set_facecolor(color)
    
    axes[1, 0].set_title('情感强度分布箱线图')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 情感相关性热力图
    correlation_matrix = emotion_data.corr()
    im = axes[1, 1].imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    axes[1, 1].set_xticks(range(len(correlation_matrix.columns)))
    axes[1, 1].set_yticks(range(len(correlation_matrix.columns)))
    axes[1, 1].set_xticklabels(correlation_matrix.columns, rotation=45)
    axes[1, 1].set_yticklabels(correlation_matrix.columns)
    axes[1, 1].set_title('情感相关性热力图')
    
    # 添加颜色条
    plt.colorbar(im, ax=axes[1, 1])
    
    # 在热力图上添加数值
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            axes[1, 1].text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                           ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    plt.savefig('old_man_sea_emotion_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# 生成详细报告
def generate_emotion_report(result_df, total_stats, peak_emotions):
    """
    生成详细的情感分析报告
    """
    print("\n" + "="*80)
    print("《老人与海》情感分析详细报告")
    print("="*80)
    
    print(f"\n总体分析统计:")
    print(f"- 分析文本块数量: {len(result_df)}")
    print(f"- 总词汇量: {result_df['word_count'].sum()} 单词")
    print(f"- 平均每个文本块词汇量: {result_df['word_count'].mean():.1f} 单词")
    
    print(f"\n情感强度排名:")
    emotion_totals = {emotion: total_stats[f'total_{emotion.lower()}'] 
                     for emotion in ['Positive', 'Negative', 'Joy', 'Sadness', 'Fear', 
                                   'Anger', 'Anticipation', 'Surprise', 'Trust', 'Disgust']}
    sorted_emotions = sorted(emotion_totals.items(), key=lambda x: x[1], reverse=True)
    
    for i, (emotion, total) in enumerate(sorted_emotions, 1):
        print(f"{i:2d}. {emotion:12s}: {total:8.1f} (平均每块: {total/len(result_df):.2f})")
    
    print(f"\n情感高峰段落分析:")
    for emotion in ['Positive', 'Negative', 'Joy', 'Sadness', 'Fear']:
        info = peak_emotions[emotion]
        if info['value'] > 0:
            print(f"\n{emotion}情感高峰 - 块 {info['chunk_id']}:")
            print(f"   强度: {info['value']:.2f}")
            print(f"   内容预览: {info['text_preview']}")

# 主执行函数
def main():
    # 文件路径 - 使用您提供的路径
    novel_file_path = r"C:\Users\11735\Downloads\The Old Man and the Sea (Ernest Hemingway) (Z-Library).txt"
    
    print("开始分析《老人与海》情感...")
    print(f"文件路径: {novel_file_path}")
    
    # 读取小说文本
    novel_text = read_old_man_and_sea(novel_file_path)
    
    if novel_text is None:
        print("无法读取文件，程序退出。")
        return
    
    # 进行情感分析（每40个句子作为一个分析块）
    print("\n开始情感分析，这可能需要一些时间...")
    results = text_emotion_large_text(novel_text, chunk_size=40)
    
    if results is None:
        print("情感分析失败，请检查情感词典文件路径。")
        return
    
    # 保存结果到CSV文件
    output_file = "old_man_sea_emotion_analysis.csv"
    results.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n分析完成！结果已保存到: {output_file}")
    
    # 显示基本结果
    print("\n前3个分析块的结果:")
    print(results.head(3))
    
    # 生成统计分析
    total_stats, peak_emotions = analyze_novel_emotion_trends(results)
    
    # 生成详细报告
    generate_emotion_report(results, total_stats, peak_emotions)
    
    # 可视化结果
    print("\n生成可视化图表...")
    visualize_emotion_trends(results)
    
    print("\n" + "="*80)
    print("分析完成！")
    print(f"1. 数据文件: {output_file}")
    print("2. 可视化图表: old_man_sea_emotion_analysis.png")
    print("3. 请在上述报告中查看详细分析结果")
    print("="*80)

# 如果直接运行此脚本，执行主函数
if __name__ == "__main__":
    main()
