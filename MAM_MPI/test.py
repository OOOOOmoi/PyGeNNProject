#!/usr/bin/env python3
import os
import sys
from pathlib import Path

def count_lines(file_path):
    """统计文件的代码行、注释行、空行"""
    total_lines = 0
    code_lines = 0
    comment_lines = 0
    blank_lines = 0
    
    # 根据文件扩展名确定注释风格
    ext = file_path.suffix.lower()
    
    # 定义不同语言的注释模式
    comment_patterns = {
        '.py': {'single': '#', 'multi_start': '"""', 'multi_end': '"""'},
        '.java': {'single': '//', 'multi_start': '/*', 'multi_end': '*/'},
        '.cpp': {'single': '//', 'multi_start': '/*', 'multi_end': '*/'},
        '.c': {'single': '//', 'multi_start': '/*', 'multi_end': '*/'},
        '.js': {'single': '//', 'multi_start': '/*', 'multi_end': '*/'},
        '.ts': {'single': '//', 'multi_start': '/*', 'multi_end': '*/'},
        '.go': {'single': '//', 'multi_start': '/*', 'multi_end': '*/'},
        '.rs': {'single': '//', 'multi_start': '/*', 'multi_end': '*/'},
        '.sh': {'single': '#', 'multi_start': '', 'multi_end': ''},
    }
    
    patterns = comment_patterns.get(ext, {'single': '', 'multi_start': '', 'multi_end': ''})
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            in_multi_comment = False
            
            for line in f:
                total_lines += 1
                line_stripped = line.strip()
                
                # 检查空行
                if not line_stripped:
                    blank_lines += 1
                    continue
                
                # 检查多行注释
                if patterns['multi_start'] and patterns['multi_start'] in line_stripped:
                    if patterns['multi_end'] in line_stripped and line_stripped.index(patterns['multi_start']) < line_stripped.index(patterns['multi_end']):
                        comment_lines += 1  # 单行多行注释
                    else:
                        in_multi_comment = True
                        comment_lines += 1
                elif in_multi_comment:
                    comment_lines += 1
                    if patterns['multi_end'] and patterns['multi_end'] in line_stripped:
                        in_multi_comment = False
                # 检查单行注释
                elif patterns['single'] and line_stripped.startswith(patterns['single']):
                    comment_lines += 1
                else:
                    code_lines += 1
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return {
        'total': total_lines,
        'code': code_lines,
        'comment': comment_lines,
        'blank': blank_lines
    }

def calculate_comment_rate(directory, extensions=None):
    """计算目录下所有文件的注释率"""
    if extensions is None:
        extensions = ['.py', '.java', '.cpp', '.c', '.js', '.ts', '.go', '.rs']
    
    total_stats = {'code': 0, 'comment': 0, 'blank': 0, 'total': 0}
    
    for root, dirs, files in os.walk(directory):
        # 忽略一些常见目录
        ignore_dirs = ['.git', '__pycache__', 'node_modules', 'vendor', 'target']
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() in extensions:
                stats = count_lines(file_path)
                for key in total_stats:
                    total_stats[key] += stats[key]
    
    if total_stats['code'] + total_stats['comment'] > 0:
        comment_rate = total_stats['comment'] / (total_stats['code'] + total_stats['comment']) * 100
    else:
        comment_rate = 0
    
    return total_stats, comment_rate

if __name__ == "__main__":
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = "."
    
    stats, rate = calculate_comment_rate(directory)
    
    print(f"代码行数: {stats['code']}")
    print(f"注释行数: {stats['comment']}")
    print(f"空行数: {stats['blank']}")
    print(f"总行数: {stats['total']}")
    print(f"注释率: {rate:.2f}%")
    print(f"注释密度: {stats['comment'] / max(stats['code'], 1):.2f} (注释行/千代码行)")